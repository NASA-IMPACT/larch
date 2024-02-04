#!/usr/bin/env python3

import json
import os
from typing import Callable, List, Optional, Type

import instructor
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ._base import AbstractMetadataExtractor


class SimpleOpenAIMetadataExtractor(AbstractMetadataExtractor):
    """
    Barebone extractor that just hits OpenAI API based on instructor patching.
    No function calling.
    TODO:
        Bugfix nasty side-effects because of instructor patch.
        Calling this 2nd time after we called `InstructorBasedOpenAIMetadataExtractor`
        throws validation error.
    """

    _SYSTEM_PROMPT = (
        "Extract metadata and entities details accurately from my requests."
    )

    def __init__(
        self,
        schema: Type[BaseModel],
        model: str = "gpt-3.5-turbo",
        openai_client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
        system_prompt: str = _SYSTEM_PROMPT,
        preprocessor: Optional[Callable] = None,
        mode: instructor.function_calls.Mode = instructor.function_calls.Mode.FUNCTIONS,
        max_retries: int = 1,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug, preprocessor=preprocessor)
        self.model = model
        self.schema = schema
        self._system_prompt = system_prompt
        self.max_retries = max_retries
        self.openai_client = openai_client or OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self.mode = mode or instructor.function_calls.mode.FUNCTIONS

    def _get_messages(self, text: str) -> List[dict]:
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        messages.append({"role": "user", "content": text})

        return messages

    def _extract(self, text: str):
        text = text.strip()
        client = instructor.patch(self.openai_client.copy(), mode=self.mode)
        messages = self._get_messages(text)
        if self.debug:
            logger.debug(f"messages :: {messages}")
        metadata = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_model=self.schema,
            max_retries=self.max_retries,
            messages=messages,
        )
        return metadata


class InstructorBasedOpenAIMetadataExtractor(SimpleOpenAIMetadataExtractor):
    """
    This uses `instructor` to hit the OpenAI function calling api.
    Note: The schema for metadata should be of `Type[OpenAISchema]`.
    """

    def _extract(self, text: str):
        if self.debug:
            logger.debug(f"nchars={len(text)}\nText :: {text}")
        schema = instructor.openai_schema(self.schema)
        print(schema.openai_schema)
        response = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=0,
            functions=[schema.openai_schema],
            function_call={"name": schema.openai_schema["name"]},
            messages=self._get_messages(text),
        )
        if self.debug:
            logger.debug(response)

        result = self.schema.model_construct()

        try:
            result = schema.from_response(response, mode=self.mode)
        except ValidationError:
            logger.warning("Bypassing validation error!")
            message = response["choices"][0]["message"]
            result = (
                self.schema.model_construct(
                    **json.loads(message["function_call"]["arguments"]),
                )
                if "function_call" in message
                else result
            )
        return result


def main():
    pass


if __name__ == "__main__":
    main()
