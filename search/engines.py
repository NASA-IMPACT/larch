#!/usr/bin/env python3

import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Type

import openai
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX
from langchain.agents.agent_types import AgentType
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import create_sql_query_chain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.utilities import SQLDatabase
from loguru import logger

from ..indexing import DocumentIndexer
from ..metadata import AbstractMetadataExtractor
from ..prompts import QA_DOCUMENTS_PROMPT, SQL_AGENT_QUERY_AUGMENTATION_PROMPT
from ..structures import Document, LangchainDocument, Response
from ..utils import remove_nulls


class AbstractSearchEngine(ABC):
    """
    Abstraction type for search engine.
    Each implemented subclass should implement `query(...)` method.
    Can be used as a callable.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @abstractmethod
    def query(self, query: str, **kwargs) -> Response:
        raise NotImplementedError()

    def search(self, query: str, **kwargs) -> Response:
        return self.query(query, **kwargs)

    def __call__(self, *args, **kwargs) -> Response:
        return self.query(*args, **kwargs)

    @property
    def __classname__(self) -> str:
        return self.__class__.__name__


class SimpleRAG(AbstractSearchEngine):
    """
    This uses vector store and indexed document for full text search.
    This is appropriate when the input query/question does not conform
    to the metadata directly. Use this when all other tools fail.
    """

    def __init__(
        self,
        document_indexer: Type[DocumentIndexer],
        cache: bool = True,
    ) -> None:
        self.document_indexer = document_indexer
        self._cache = cache
        self.cache_store = {}

    def query(self, query: str, top_k: int = 5, **kwargs) -> Response:
        query_hash = hash(query)
        result = self.cache_store.get(query_hash) or self.document_indexer.query(
            query,
            top_k=top_k,
            **kwargs,
        )
        result.source = (
            f"{self.__classname__}.{self.document_indexer.__class__.__name__}"
        )
        self.cache_store[query_hash] = result
        return result


class InMemoryDocumentQAEngine(AbstractSearchEngine):
    """
    This QA Engine uses provided list of documents to answer questions
    based on only those documents.
    """

    _DEFAULT_NO_RESPONSE = "No answer can be inferred!"

    def __init__(
        self,
        documents: List[Document],
        llm: Optional[BaseLanguageModel] = None,
        prompt: Optional[BasePromptTemplate] = QA_DOCUMENTS_PROMPT,
        combine_technique: "str" = "stuff",
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.llm = llm or ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
        self.documents = documents or []
        self.prompt = prompt
        self.combine_technique = (combine_technique or "stuff").strip()
        self.chain = self.get_chain(self.combine_technique)

    def get_chain(self, combine_technique: str = "stuff") -> LLMChain:
        if combine_technique == "stuff":
            return load_qa_chain(
                llm=self.llm,
                chain_type=combine_technique,
                prompt=self.prompt,
                verbose=self.debug,
            )
        return load_qa_chain(
            llm=self.llm,
            chain_type=combine_technique,
            verbose=self.debug,
        )

    def query(self, query: str, **kwargs) -> Response:
        documents = self.documents or kwargs.get("documents", [])
        if not documents:
            logger.warning("Empty document list!")
            return Response(text=self._DEFAULT_NO_RESPONSE, source=self.__classname__)
        docs = list(map(lambda x: x.as_langchain_document(), documents))
        res = self.chain(
            dict(input_documents=docs, question=query),
            return_only_outputs=True,
        ).get("output_text", "")
        return Response(text=res, evidences=documents, source=self.__classname__)


class DocumentStoreRAG(AbstractSearchEngine):
    """
    This RAG uses provided document indexer/store to first fetch/retrieve
    top_k documents which are used for further augmented generation
    based on given query.
    """

    def __init__(
        self,
        document_store: DocumentIndexer,
        qa_engine: Optional[InMemoryDocumentQAEngine] = None,
        llm: Optional[BaseLanguageModel] = None,
        prompt: Optional[BasePromptTemplate] = QA_DOCUMENTS_PROMPT,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.document_store = document_store

        # either one of them has to be provided
        if qa_engine is None and llm is None:
            raise ValueError(
                "Both the qa_engine and llm arguments can't be none. Expected: provide at least one of them!",
            )

        self.qa_engine = qa_engine or InMemoryDocumentQAEngine(
            documents=None,
            llm=llm,
            prompt=prompt,
            debug=debug,
        )

    @property
    def llm(self):
        return self.qa_engine.llm

    @property
    def prompt(self):
        return self.qa_engine.prompt

    def query(self, query: str, top_k: int = 5, **kwargs) -> Response:
        documents = self.document_store.query_top_k(query=query, top_k=top_k, **kwargs)
        result = self.qa_engine(query=query, documents=documents, **kwargs)
        result.source = f"{self.__classname__}.{self.document_store.__class__.__name__}"
        if self.debug:
            logger.debug(f"top_k={top_k} documents :: {documents}")
            logger.debug(f"Result={result}")
        return result


class SQLAgentSearchEngine(AbstractSearchEngine):
    """
    This search engine uses an SQL agent to best answer the given query.
    Based on given query, it generates an appropriate SQL query internally and
    runs the query to generate response text. Use this whenever there's a
    complex query that SQL syntax can support such as grouping, ordering,
    counting, etc.

    Args:
        db_uri (str): Database URI to connect to the database
        tables (list): List of tables to include in the database
        llm (BaseLanguageModel): Language model to use for SQL agent
        prompt_prefix (str): Prompt prefix to use for SQL agent
        sql_fuzzy_threshold (float): Threshold to use for SQL agent
        query_augmentation_prompt (str): Prompt to use for query augmentation
        railguard_response (bool): railguard response or not
        debug (bool): Debug mode
    """

    _RESTRICTED_KEYWORDS = [
        "table_information",
        "Let's query the schema",
        "Here is the query",
        "The tables in the database",
        "Here is the sql query",
    ]

    _DEFAULT_RESPONSE = "Answer could not be found."

    _SQL_QUERY_PATTERN = re.compile(
        r"SELECT \* FROM \w+ WHERE .*",
        re.IGNORECASE,
    )

    def __init__(
        self,
        db_uri: str,
        tables: Optional[List[str]] = None,
        llm: Optional[BaseLanguageModel] = None,
        prompt_prefix: Optional[str] = None,
        sql_fuzzy_threshold: float = 0.75,
        query_augmentation_prompt: Optional[str] = None,
        railguard_response: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.query_augmentation_prompt = (
            query_augmentation_prompt or SQL_AGENT_QUERY_AUGMENTATION_PROMPT
        )

        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        self.db = SQLDatabase.from_uri(db_uri, include_tables=tables)

        self.prompt_prefix = prompt_prefix or SQL_PREFIX
        self.sql_fuzzy_threshold = sql_fuzzy_threshold
        self.railguard_response = railguard_response
        self.tables = tables

        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=OpenAI(temperature=0)),
            verbose=self.debug,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=self.prompt_prefix,
        )

    @staticmethod
    def augment_query(query: str, augmentation_prompt: str, threshold: float) -> str:
        augmentation_prompt = (
            augmentation_prompt.format(threshold=threshold)
            if "{threshold}" in augmentation_prompt
            else augmentation_prompt
        )
        return f"Query: {query}\n" + augmentation_prompt

    def _check_restricted_keywords(self, response: str) -> bool:
        """
        _check_restricted_keywords is a method to check if the response
        has restricted keywords or not.

        Args:
            response (str): response text

        Returns:
            bool: True if restricted keywords are found, False otherwise
        """
        restricted_keywords_pattern = re.compile(
            r"\b(" + "|".join(self._RESTRICTED_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )

        matched_restricted_keywords = restricted_keywords_pattern.search(response)
        if self.debug:
            logger.debug(f"Matched restricted keywords: {matched_restricted_keywords}")

        return bool(matched_restricted_keywords)

    def _check_sql_query(self, response: str) -> bool:
        """
        _check_sql_query is a method to check if the response
        has SQL query statement in it or not.

        Args:
            response (str): response text

        Returns:
            bool: True if sql query statement is found, False otherwise
        """
        match_sql_query = self._SQL_QUERY_PATTERN.search(response)
        if self.debug:
            logger.debug(f"Matched SQL query: {match_sql_query}")
        return bool(match_sql_query)

    def _check_table_information(self, response: str) -> bool:
        """
        _check_table_information is a method to check if the response
        is leaking table information or not.

        Args:
            response (str): response text

        Returns:
            bool: True if table information is found, False otherwise
        """
        tables = self.tables or self.db.get_usable_table_names()

        table_text = (
            ", ".join(f"{table}" for table in tables[:-1])
            + (", and " if len(tables) > 1 else "")
            + f"{tables[-1]}"
        )
        tables_pattern = re.compile(rf"{table_text}", re.IGNORECASE)

        match = tables_pattern.search(response.replace("`", ""))
        if self.debug:
            logger.debug(
                f"Matched table information: {match} with pattern: {tables_pattern}",
            )

        return bool(match)

    def prevent_response_leakage(self, response: str) -> str:
        """
        prevent_response_leakage is a method to railguard the
        agent from leaking table information.

        Args:
            response (str): response text

        Returns:
            str: railguarded response
        """
        if (
            self._check_restricted_keywords(
                response,
            )
            or self._check_sql_query(
                response,
            )
            or self._check_table_information(
                response,
            )
        ):
            if self.debug:
                logger.debug(
                    """Restricted information
                             found in response. Railguarding...""",
                )
            return self._DEFAULT_RESPONSE

        return response

    def query(self, query: str, **kwargs) -> Response:

        query = SQLAgentSearchEngine.augment_query(
            query,
            self.query_augmentation_prompt,
            self.sql_fuzzy_threshold,
        )
        result = self.agent_executor.run(query)
        if self.debug:
            logger.debug(f"Query = {query}")
            logger.debug(f"Result={result}")

        if self.railguard_response:
            try:
                result = self.prevent_response_leakage(result)
            except Exception as e:
                logger.warning(
                    f"Error while railguarding response for query: {query} with error: {e}",
                )

        return Response(text=result, source=self.__classname__)


class MetadataBasedAugmentedSearchEngine(AbstractSearchEngine):
    """
    It uses text-to-sql algorithm through langchain's sql-query-chain.
    It first extracts metadata from the given input query using metadata
    extractor. Then uses the extracted json to best construct a valid
    SQL query which is then further executed on the database to return
    the final answer. The best case scenario to use this is when we
    have sufficiently enough metadta information to generate a very
    accurate SQL query.
    """

    _SQLITE_PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        output_parser=None,
        partial_variables={},
        template='You are an expert in SQL. Given an input metadata json extracted from an input query, first create a syntactically correct SQLite query to run on the table below, then look at the results of the query and return the answer to the input question.\nUnless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.\nNever query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.\nPay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.\nUse LIKE clause in the SQL query for string matching.\nPay attention to use date(\'now\') function to get the current date, if the question involves "today".\n\nUse the following format:\n\nQuestion: metadata JSON here\nSQLQuery: Correctly extracted SQL Query to run\nSQLResult: Result of the SQLQuery\nAnswer: Final answer here\n\nOnly use the following tables:\n{table_info}\n\nQuestion: {input}',
        template_format="f-string",
        validate_template=True,
    )

    def __init__(
        self,
        metadata_extractor: Type[AbstractMetadataExtractor],
        db_uri: str,
        model: str = "gpt-3.5-turbo-0613",
        tables: Optional[List[str]] = None,
        prompt_template: Optional[PromptTemplate] = None,
        bypass_metadata_extractor: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.metadata_extractor = metadata_extractor
        self.metadata_extractor.debug = self.debug
        self.bypass_metadata_extractor = bypass_metadata_extractor

        self.prompt = prompt_template or (
            MetadataBasedAugmentedSearchEngine._SQLITE_PROMPT
            if not bypass_metadata_extractor
            else None
        )

        self.db = SQLDatabase.from_uri(db_uri, include_tables=tables)
        self.llm = ChatOpenAI(temperature=0.0, model=model)
        self.db_chain = create_sql_query_chain(self.llm, self.db, self.prompt)

    def query(self, query: str, top_k: int = 5, **kwargs) -> Response:
        db_chain = create_sql_query_chain(self.llm, self.db, self.prompt, top_k)
        metadata = {}
        if not self.bypass_metadata_extractor:
            metadata = self.metadata_extractor(query)
            metadata = remove_nulls(metadata.model_dump())
            query = f"Query: {query}\nmetadata: {json.dumps(metadata)}"
        sql_query = db_chain.invoke(dict(question=query))

        if self.debug:
            logger.debug(f"{query}")
            logger.debug(f"SQL Query: {sql_query}")

        res = ""
        try:
            res = self.db.run(sql_query)
        except:
            logger.warning(f"Error in {self.__class__.__name__}")
        return Response(
            text=res,
            source=self.__classname__,
            extras=dict(metadata=metadata),
            evidences=[Document(text=sql_query, source="create_sql_query_chain")],
        )


class EnsembleAugmentedSearchEngine(AbstractSearchEngine):
    """
    This ensembles answers from all the available search engines using
    simple prompting strategy for GPT
    """

    _PROMPT = """
    Following are the systems generated answers in response to user input query.
    I want you to best consolidate the response that is accurate and coherent based on
    all those answers. Don't give response that doesn't belong within the generated
    answers. Here are the answers:\n
    """

    def __init__(
        self,
        *engines: Type[AbstractSearchEngine],
        model: str = "gpt-3.5-turbo",
        prompt: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(debug)
        self.engines = engines
        self.prompt = prompt or EnsembleAugmentedSearchEngine._PROMPT

        for engine in self.engines:
            engine.debug = self.debug

        self.model = model or "gpt-3.5-turbo"

    def query(self, query: str, **kwargs) -> Response:
        results = map(lambda e: e(query, **kwargs), self.engines)
        results = list(results)

        msg = self.prompt + "\n".join(
            [f"Answer {i}: {r}" for i, r in enumerate(results, start=1)],
        )

        if self.debug:
            logger.debug(f"User Message :: {msg}")

        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": msg}],
        )

        if self.debug:
            logger.debug(f"OpenAI response :: {response}")

        response = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return Response(text=response, source=self.__classname__, evidences=results)


class MultiRetrieverSearchEngine(AbstractSearchEngine):
    """
    This aggregates answers from all the available sources
    using multiple retriever model.
    """

    _SYSTEM_PROMPT = (
        "You are very accurate information retriever."
        + " Use the information from the sources below to answer any query."
        + " Only answer based on the provided sources."
        + " Try best to consolidate the answer that is accurate"
        + " and coherent based on the answers from all the sources."
        + " Don't give response that doesn't belong within the generated answers."
        + " Be as concise as possible to give the final answer."
        # + " If there are conflicting information from all the sources,"
        # + " use the answer from SQLAgentSearchEngine as high priority if possible."
    )

    class RetrieverWrapper(BaseRetriever):
        search_engine: AbstractSearchEngine
        top_k: int = 5
        response: Optional[Response] = None

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            **kwargs,
        ) -> List[LangchainDocument]:
            self.response = self.search_engine(query, top_k=self.top_k, **kwargs)
            return self.response.as_langchain_document(ignore_extras=True)

    def __init__(
        self,
        *engines: AbstractSearchEngine,
        llm: Optional[BaseLanguageModel] = None,
        system_prompt: str = _SYSTEM_PROMPT,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.engines = engines
        self.llm = llm or ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

        self._source_mapping = {
            f"Source {i}": engine.__classname__
            for i, engine in enumerate(self.engines, start=1)
        }

        # TODO: Functionalize
        retriever_prompt = self._construct_system_prompt(*engines)
        system_prompt = system_prompt or MultiRetrieverSearchEngine._SYSTEM_PROMPT
        system_prompt = f"{system_prompt}\n{retriever_prompt}"
        self.system_prompt = system_prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{question}")],
        )

    @staticmethod
    def _get_retrievers(
        *engines: AbstractSearchEngine,
        top_k: int = 5,
    ) -> List[RetrieverWrapper]:
        return list(
            map(
                lambda engine: MultiRetrieverSearchEngine.RetrieverWrapper(
                    search_engine=engine,
                    top_k=top_k,
                ),
                engines,
            ),
        )

    def _construct_system_prompt(self, *engines: AbstractSearchEngine):
        res = ""

        for i, engine in enumerate(engines, start=1):
            name = self._source_mapping[f"Source {i}"]
            res += f"Source {i}: {name}\n<source{i}>\n{{source{i}}}\n</source{i}>\n"
        return res

    def _postprocess(self, text: str) -> str:
        def _replace_source(match):
            key = match.group(1)
            return match.group(0).replace(key, self._source_mapping.get(key, key))

        return re.sub(r"\s+(Source \d)", _replace_source, text).strip()

    @staticmethod
    def _build_chain(*retrievers, prompt, llm):
        retriever_chain = {
            f"source{i}": (lambda x: x["question"]) | retriever
            for i, retriever in enumerate(retrievers, start=1)
        }
        return (
            {**retriever_chain, **{"question": lambda x: x["question"]}} | prompt | llm
        )

    def query(self, query: str, top_k: int = 5, **kwargs) -> Response:
        # retrievers are made stateful to cache responses
        # So, need to build at runtime
        retrievers = self._get_retrievers(*self.engines, top_k=top_k)

        chain = self._build_chain(*retrievers, prompt=self.prompt, llm=self.llm)

        result = chain.invoke(dict(question=query))
        if self.debug:
            logger.debug(result)

        return Response(
            text=self._postprocess(result.content),
            source=self.__classname__,
            evidences=list(map(lambda r: r.response, retrievers)),
        )


def main():
    pass


if __name__ == "__main__":
    main()
