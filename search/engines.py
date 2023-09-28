#!/usr/bin/env python3

import json
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chains import create_sql_query_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import SQLDatabase
from loguru import logger

from ..indexing import DocumentIndexer
from ..metadata import AbstractMetadataExtractor


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
    This is appropriate when the input query/question does not conform to the metadata directly.
    Use this when all other tools fail.
    """

    def __init__(
        self,
        document_indexer: Type[DocumentIndexer],
        cache: bool = True,
    ) -> None:
        self.document_indexer = document_indexer
        self._cache = cache
        self.cache_store = {}

    def query(self, query: str) -> str:
        query_hash = hash(query)
        result = self.cache_store.get(query_hash) or self.document_indexer.query(query)
        self.cache_store[query_hash] = result
        return result


class SQLAgentSearchEngine(AbstractSearchEngine):
    """
    This search engine uses langchain's SQL agent to best answer the given query.
    Based on given query, it generates an appropriate SQL query internally and runs the query to generate response text.

    This should be the first engine to be run if there is enough metadata.
    If it fails to answer the query, use `MetadataBasedAugmentedSearchEngine` instead.
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
    It first extracts metadata from the given input query using metadata extractor.
    Then uses the extracted json to best construct a valid SQL query which is then further executed
    on the database to return the final answer.
    The best case scenario to use this is when we have sufficiently enough metadta information to generate a very accurate SQL query.
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


def main():
    pass


if __name__ == "__main__":
    main()
