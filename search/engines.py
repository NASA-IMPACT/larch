#!/usr/bin/env python3

import json
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

import openai
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.base_language import BaseLanguageModel
from langchain.chains import create_sql_query_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.utilities import SQLDatabase
from loguru import logger
from pynequa import QueryParams, Sinequa

from ..indexing import DocumentIndexer
from ..metadata import AbstractMetadataExtractor
from ..prompts import QA_DOCUMENTS_PROMPT
from ..structures import Document


class AbstractSearchEngine(ABC):
    """
    Abstraction type for search engine.
    Each implemented subclass should implement `query(...)` method.
    Can be used as a callable.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @abstractmethod
    def query(self, query: str, **kwargs) -> str:
        raise NotImplementedError()

    def search(self, query: str, **kwargs) -> str:
        return self.query(query, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self.query(*args, **kwargs)


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

    def query(self, query: str, **kwargs) -> str:
        query_hash = hash(query)
        result = (
            self.cache_store.get(query_hash)
            or self.document_indexer.query(query, **kwargs).strip()
        )
        self.cache_store[query_hash] = result
        return result


class InMemoryDocumentQAEngine(AbstractSearchEngine):
    """
    This QA Engine uses provided list of documents to answer questions
    based on only those documents.
    """

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
        self.chain = load_qa_chain(
            llm=self.llm,
            chain_type=combine_technique,
            prompt=prompt,
            verbose=self.debug,
        )

    def query(self, query: str, **kwargs) -> str:
        documents = self.documents or kwargs.get("documents", [])
        if not documents:
            logger.warning("Empty document list!")
            return ""
        docs = list(map(lambda x: x.as_langchain_document(), documents))
        return self.chain(
            dict(input_documents=docs, question=query),
            return_only_outputs=True,
        ).get("output_text", "")


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
        self.qa_engine = qa_engine or InMemoryDocumentQAEngine(
            documents=None,
            llm=llm,
            prompt=prompt,
            debug=debug,
        )
        self.llm = llm
        self.prompt = prompt

    def query(self, query: str, top_k: int = 5, **kwargs) -> str:
        documents = self.document_store.query_top_k(query=query, top_k=top_k, **kwargs)
        if self.debug:
            logger.debug(f"top_k={top_k} documents :: {documents}")
        return self.qa_engine(query=query, documents=documents, **kwargs)


class SQLAgentSearchEngine(AbstractSearchEngine):
    """
    This search engine uses an SQL agent to best answer the given query.
    Based on given query, it generates an appropriate SQL query internally and
    runs the query to generate response text. Use this whenever there's a
    complex query that SQL syntax can support such as grouping, ordering,
    counting, etc. If this fails to answer the query, use
    `MetadataBasedAugmentedSearchEngine` instead.
    """

    def __init__(
        self,
        db_uri: str,
        tables: list,
        llm: Optional = None,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        self.db = SQLDatabase.from_uri(db_uri, include_tables=tables)

        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=OpenAI(temperature=0)),
            verbose=self.debug,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

    def query(self, query: str, **kwargs) -> str:
        return self.agent_executor.run(query)


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
        template='You are a SQLite expert. Given an input metadata json extracted from an input query, first create a syntactically correct SQLite query to run on the table below, then look at the results of the query and return the answer to the input question.\nUnless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.\nNever query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.\nPay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.\nUse LIKE clause in the SQL query for string matching.\nPay attention to use date(\'now\') function to get the current date, if the question involves "today".\n\nUse the following format:\n\nQuestion: metadata JSON here\nSQLQuery: Correctly extracted SQL Query to run\nSQLResult: Result of the SQLQuery\nAnswer: Final answer here\n\nOnly use the following tables:\n{table_info}\n\nQuestion: {input}',
        template_format="f-string",
        validate_template=True,
    )

    def __init__(
        self,
        metadata_extractor: Type[AbstractMetadataExtractor],
        db_uri: str,
        model: str = "gpt-3.5-turbo-0613",
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

        self.db = SQLDatabase.from_uri(db_uri, include_tables=["metadata"])
        self.llm = ChatOpenAI(temperature=0.0, model=model)
        self.db_chain = create_sql_query_chain(self.llm, self.db, self.prompt)

    def clean_metadata(self, metadata: dict) -> dict:
        res = {}
        for k, v in metadata.items():
            if v:
                res[k] = v
        return res

    def query(self, query: str, top_k: int = 5) -> str:
        db_chain = create_sql_query_chain(self.llm, self.db, self.prompt, top_k)
        if not self.bypass_metadata_extractor:
            metadata = self.metadata_extractor(query)
            metadata = self.clean_metadata(metadata.dict_flattened())
            query = f"Query: {query}\nmetadata: {json.dumps(metadata)}"
        sql_query = db_chain.invoke(dict(question=query))

        if self.debug:
            logger.debug(f"{query}")
            logger.debug(f"SQL Query: {sql_query}")

        return self.db.run(sql_query)


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

    def query(self, query: str, **kwargs) -> str:
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

        return response.get("choices", [{}])[0].get("message", {}).get("content", "")


class SinequaSearchEngine(AbstractSearchEngine):
    """
    This class uses Sinequa as search engine. Given a query and metadata
    it will perform faceted searcha and provide a set of matches.

    Args:
        base_url (str): The base URL for the sinequa instance
        auth_token (str): Authentication token for sinequa instance
        app_name (str, optional): The name of the Sinequa application
                                (default is "vanilla-search").
        query_name (str, optional): The name of the query (default
                                is "query").
        debug (bool, optional): Flag indicating whether to enable debug mode (default is False).
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        app_name: str = "vanilla-search",
        query_name: str = "query",
        debug: bool = False,
    ) -> None:
        """
        Initializes a new instance of SinequaSearchEngine class.
        """
        super().__init__(debug)
        self.sinequa = Sinequa.from_config(
            cfg={
                "base_url": base_url,
                "access_token": auth_token,
                "app_name": app_name,
                "query_name": query_name,
            },
        )

    def query(self, query: str, **kwargs) -> str:
        """ "
        Executes a search query on the Sinequa platform

        Args:
            query (str): The search query.
            **kwargs: Additional keyword arguments (e.g., page, page_size).

        Returns:
            str: The matching passages as a single string.
        """
        # build query params payload

        if self.debug:
            logger.debug(f"query: {query}")

        query_params = QueryParams()
        query_params.search_text = query
        query_params.page = kwargs.get("page", 1)
        query_params.page_size = kwargs.get("page_size", 10)

        results = self.sinequa.search_query(query_params)

        if "ErrorCode" in results:
            error_message = results["ErrorMessage"]
            logger.error(error_message)
            raise Exception(f"Error: {error_message}")

        # the answer we're looking for is inside
        # topPassages -> passages -> [highlightedText]

        top_passages = results["topPassages"]["passages"]

        matching_passages = ""
        for passages in top_passages:
            matching_passages += passages["highlightedText"] + " "

        return matching_passages


def main():
    pass


if __name__ == "__main__":
    main()
