#!/usr/bin/env python3

import json
from typing import Callable, Dict, List, Optional, Type, Union

from joblib import Parallel, delayed
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
    RetryWithErrorOutputParser,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BasePromptTemplate, OutputParserException
from langchain_core.language_models import BaseLanguageModel
from loguru import logger
from pydantic import BaseModel

from ..utils import PydanticOutputParserWithoutValidation, get_cpu_count
from ._base import AbstractMetadataExtractor, MetadataAggregator
from .chunker import InstructorAggregator, TokenChunker


class LangchainBasedMetadataExtractor(AbstractMetadataExtractor):
    """
    This uses barebone langchain's extraction chain.
    """

    def __init__(
        self,
        schema: Type[BaseModel],
        llm: Optional[Type[BaseLanguageModel]] = None,
        prompt: Optional[BasePromptTemplate] = None,
        preprocessor: Optional[Callable] = None,
        debug: bool = False,
    ):
        super().__init__(debug=debug, preprocessor=preprocessor)
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
        output_parser: Optional[PydanticOutputParser] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(preprocessor=preprocessor, debug=debug)
        self.schema = schema
        self.parser = output_parser or PydanticOutputParserWithoutValidation(
            pydantic_object=schema,
        )
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
        if self.debug:
            logger.debug(f"Model response :: {response}")
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


class ChunkBasedMetadataExtractor(AbstractMetadataExtractor):
    """
    Takes large input texts, chunks them, and then passes them in parallel to OpenAI.
    This results in a list of metadata extractions, which is then aggregated.
    """

    _MAX_JOBS = get_cpu_count()

    def __init__(
        self,
        extractor: Type[AbstractMetadataExtractor],
        chunker: Optional = None,
        aggregator: Type[MetadataAggregator] = None,
        n_jobs: int = 1,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.extractor = extractor
        self.chunker = chunker or TokenChunker()
        self.aggregator = aggregator or InstructorAggregator(
            extractor.schema,
        )  # assumes each extractor would have schema

        n_jobs = min(max(1, n_jobs), ChunkBasedMetadataExtractor._MAX_JOBS)
        self.n_jobs = n_jobs

    def _extract_in_parallel(self, chunks: List[str]) -> List[Type[BaseModel]]:
        """Takes a list of string chunks and runs an extractor on them in parallel"""
        res = []
        try:
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(self.extractor)(chunk) for chunk in chunks
            )
        except ValueError:
            logger.warning("Switching joblib backend to 'threading'")
            res = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.extractor)(chunk) for chunk in chunks
            )
        return res

    def _extract(self, text: str) -> Type[BaseModel]:
        chunks = self.chunker(text)
        if self.debug:
            logger.debug(f"chunk lengths | {[len(chunk) for chunk in chunks]}")
        extractions = self._extract_in_parallel(chunks)
        return self.aggregator(extractions)
