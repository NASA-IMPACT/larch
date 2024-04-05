#!/usr/bin/env python3

import json
import os
from typing import Callable, List, Optional, Type

import instructor
from loguru import logger
from openai import ChatCompletion, OpenAI
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

    def _get_messages(self, text: str) -> List[dict]:
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        messages.append({"role": "user", "content": text})

        return messages

    def _extract(self, text: str):
        text = text.strip()
        client = instructor.patch(
            self.openai_client.copy(),
            mode=instructor.function_calls.Mode.FUNCTIONS,
        )
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

    def _chat_completion(
        self,
        text: str,
        model: Optional[OpenAI] = None,
        temperature: int = 0,
        schema: Optional[Type[BaseModel]] = None,
    ) -> ChatCompletion:
        """
        A private method to access cereate ChatCompletion

        Args:
            ```text```: ```str```
                Text from which metadata is to be extracted
            ```model```: ```Optional[OpenAI]```
                An OpenAI model object. If not provided, defaults to
                what has been set in the constructor.
            ```temperature```: ```int```
                Generation parameter: temperature. Defaults to 0.
                Lower the value, more deterministic generation.
            ```schema```: ```Optional[Type[BaseModel]]```
                A pydantic class (not the object) (derived from BaseModel)
                that guides what information is to be extracted
                from the text.
        """
        schema = instructor.openai_schema(schema or self.schema)
        return self.openai_client.chat.completions.create(
            model=model or self.model,
            temperature=temperature,
            functions=[schema.openai_schema],
            function_call={"name": schema.openai_schema["name"]},
            messages=self._get_messages(text),
        )

    def _extract(self, text: str):
        if self.debug:
            logger.debug(f"nchars={len(text)}\nText :: {text}")
        schema = instructor.openai_schema(self.schema)
        response = self._chat_completion(
            text,
            temperature=0.0,
            schema=self.schema,
            model=self.model,
        )
        if self.debug:
            logger.debug(response)

        result = self.schema.model_construct()

        try:
            result = schema.from_response(
                response,
                mode=instructor.function_calls.Mode.FUNCTIONS,
            )
        except ValidationError:
            if self.debug:
                logger.warning("Bypassing validation error!")
            message = response.choices[0].message
            result = (
                self.schema.model_construct(
                    **json.loads(message.function_call.arguments),
                )
                if hasattr(message, "function_call")
                else result
            )
        return result


def main():
    pass


if __name__ == "__main__":
    main()
