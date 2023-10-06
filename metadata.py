#!/usr/bin/env python3

import contextlib
import json
import re
from abc import ABC, abstractmethod
from functools import cache
from typing import Callable, List, Optional, Type

import instructor
import openai
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseLanguageModel
from langchain.schema import BasePromptTemplate
from loguru import logger
from pydantic import BaseModel, ValidationError


class AbstractMetadataExtractor(ABC):
    """
    Each MetadataExtractor can be used like a functor through __call__.

    Any downstream implementation/subclass should implement `_extract(...)`
    method.

    Args:
        ```preprocess_func```: ```Optional[Callable]```
            A callable that preprocessing input text string.
            Defaults to merging multiple whitespace into single.
        ```debug```: ```bool```
            Flag to enable debug mode logs.
            Defaults to `False`
    """

    def __init__(
        self,
        preprocess_func: Optional[Callable] = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.preprocess_func = preprocess_func or self._preprocess_text

    def extract_metadata(self, text: str) -> Optional[Type[BaseModel]]:
        """
        Extract metadata from the input text

        Args:
            ```text```: ```str```
                Input text from which metadata is extracted

        Returns:
            Return a pydantic schema object to map the metadata
        """
        text = self._preprocess_text(text)
        return self._extract(text) if text else None

    @cache
    @abstractmethod
    def _extract(self, text: str):
        raise NotImplementedError()

    @staticmethod
    def _preprocess_text(text: str):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def __call__(self, *args, **kwargs):
        return self.extract_metadata(*args, **kwargs)


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
        model: str = "gpt-3.5-turbo-0613",
        system_prompt: str = _SYSTEM_PROMPT,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.model = model
        self.schema = schema
        self._system_prompt = system_prompt

    def _get_messages(self, text: str) -> List[dict]:
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        messages.append({"role": "user", "content": text})

        return messages

    def _extract(self, text: str):
        instructor.patch()
        metadata = openai.ChatCompletion.create(
            model=self.model,
            temperature=0,
            response_model=self.schema,
            messages=self._get_messages(text),
        )
        instructor.unpatch()
        return metadata


class InstructorBasedOpenAIMetadataExtractor(SimpleOpenAIMetadataExtractor):
    """
    This uses `instructor` to hit the OpenAI function calling api.
    Note: The schema for metadata should be of `Type[OpenAISchema]`.
    """

    def _extract(self, text: str):
        instructor.patch()
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=0,
            functions=[self.schema.openai_schema],
            function_call={"name": self.schema.openai_schema["name"]},
            messages=self._get_messages(text),
        )
        if self.debug:
            logger.debug(response)

        result = self.schema.model_construct()

        try:
            result = self.schema.from_response(response)
        except ValidationError as e:
            logger.warning("Bypassing validation error!")
            message = response["choices"][0]["message"]
            result = (
                self.schema.model_construct(
                    **json.loads(message["function_call"]["arguments"]),
                )
                if "function_call" in message
                else result
            )
        instructor.unpatch()
        return result


class LangchainBasedMetadataExtractor(AbstractMetadataExtractor):
    """
    This uses barebone langchain's extraction chain.
    """

    def __init__(
        self,
        schema: Type[BaseModel],
        llm: Optional[Type[BaseLanguageModel]] = None,
        prompt: Optional[BasePromptTemplate] = None,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.schema = schema
        self.chain = self._create_chain(schema=schema, llm=llm, prompt=prompt)

    @property
    def prompt(self):
        return self.chain.prompt

    @staticmethod
    def _create_chain(
        schema: BaseModel,
        llm: Type[BaseLanguageModel],
        prompt: Optional[BasePromptTemplate] = None,
    ):
        if not isinstance(schema, dict):
            schema = schema.model_json_schema()
        return create_extraction_chain(schema=schema, llm=llm, prompt=prompt)

    def _extract(self, text: str):
        response = self.chain.run(text)
        if self.debug:
            logger.debug(response)
        return self.schema.model_construct(**response[0]) if response else response
