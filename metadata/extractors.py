#!/usr/bin/env python3

import json
from typing import Callable, Dict, List, Optional, Type, Union

import instructor
import openai
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseLanguageModel
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
    RetryWithErrorOutputParser,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BasePromptTemplate, OutputParserException
from loguru import logger
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
        schema = instructor.openai_schema(self.schema)
        response = openai.ChatCompletion.create(
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
            result = schema.from_response(response)
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


class LegacyMetadataExtractor(AbstractMetadataExtractor):
    """
    This metadata extractor is a port of older code that uses prompting
    and schema correction to parse the response from LLM.
    """

    _PROMPT_STR = (
        "You are a helpful assistant that extracts named entities from"
        + " the given input text."
        + " You must strictly conform to the provided schema for the extraction."
        + " Do not add newline in response."
    )

    def __init__(
        self,
        schema: Type[BaseModel],
        llm: Optional[BaseLanguageModel] = None,
        prompt: Optional[Union[BasePromptTemplate, str]] = None,
        preprocessor: Optional[Callable] = None,
        whitelist_map: Optional[Dict[str, List[str]]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(preprocessor=preprocessor, debug=debug)
        self.schema = schema
        self.parser = PydanticOutputParser(pydantic_object=schema)
        self.llm = llm or ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0613")
        self.whitelist_map = whitelist_map or {}
        self.prompt = self._build_prompt(prompt)

    def _build_prompt(self, prompt) -> ChatPromptTemplate:
        if isinstance(prompt, BasePromptTemplate):
            return prompt
        prompt = prompt or LegacyMetadataExtractor._PROMPT_STR
        if isinstance(prompt, str) and "{format_instructions}" not in prompt:
            prompt = prompt + "\n{format_instructions}\n"
        if isinstance(prompt, str) and "{input}" not in prompt:
            prompt = prompt + "Input:\n{input}"
        if isinstance(prompt, str):
            prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

    def _extract_legacy(self, text: str) -> Type[BaseModel]:
        model_response = self.llm(
            self.prompt.format_prompt(
                format_instructions=self.parser.get_format_instructions(),
                input=text,
            ).to_messages(),
        )
        response = model_response.content
        response = self.__fix_response(response)
        return self.schema.model_construct(**response)

    def __fix_response(self, response: str) -> dict:
        try:
            response = self.parser.parse(response).model_dump_json()
        except OutputParserException:
            response = self.__fail_safe(response)
            response = response.model_dump_json()
        return json.loads(response)

    def __fail_safe(self, response: str) -> dict:
        if self.debug:
            logger.debug("Using fail_safe parser")
        try:
            retry_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            response = retry_parser.parse(response)
        except OutputParserException:
            retry_parser = RetryWithErrorOutputParser.from_llm(
                parser=self.parser,
                llm=self.model,
            )
            response = retry_parser.parse_with_prompt(response, self.llm)
        return response

    def _extract(self, text: str) -> Optional[Type[BaseModel]]:
        res = self._extract_legacy(text)
        return res
