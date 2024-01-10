#!/usr/bin/env python3
import ast
from typing import Callable, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from pynequa import QueryParams, Sinequa

from ..structures import Document
from ..utils import remove_duplicate_documents
from ._base import DocumentIndexer


class SinequaDocumentIndexer(DocumentIndexer):
    """
    This uses Sinequa as document store to extract metadata
    from the documents stored in Sinequa.
    """

    # not recommended to change as it might break result parsing
    columns_to_surface = [
        "text",
        "passagevectors",
        "collection",
        "treepath",
        "filename",
    ]

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        app_name: str = "vanilla-search",
        query_name: str = "query",
        columns: Optional[List[str]] = columns_to_surface,
        docs: Optional[List[str]] = None,
        text_processor: Optional[Callable] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            docs=docs,
            text_processor=text_processor,
            debug=debug,
        )

        self.sinequa = Sinequa.from_config(
            cfg={
                "base_url": base_url,
                "access_token": auth_token,
                "app_name": app_name,
                "query_name": query_name,
            },
        )
        self.columns = columns

    def _generate_sql_query(
        self,
        query: str,
        collection: str,
        limit: int = 5,
    ) -> str:
        """
        This method generates SQL query for Sinequa's SQL Engine

        Args:
            collection (str): Name of collection o query to
            query (str): query text
            limit (int): maximum number of results to return
        Returns:
            str : SQL query string
        """
        collection = f"/{collection}/*"
        column_str = ",".join(self.columns)
        return f"""SELECT {column_str} FROM index
                        WHERE collection='{collection}'
                        AND
                        text contains '{query}'
                        LIMIT {limit}"""

    def _parse_query_results(self, results) -> List[Document]:
        """
        This method parses query results from sinequa and returns
        then in a document format as a list.

        It will remove duplicate results while returning the document list.
        """

        if "ErrorCode" in results:
            error_message = results["ErrorMessage"]
            logger.error(error_message)
            raise Exception(f"Error: {error_message}")

        top_passages = results["topPassages"]["passages"]

        documents = []
        for passage in top_passages:
            matching_text = passage["highlightedText"]
            source = passage["recordId"].split("|")[1]
            documents.append(
                Document(
                    text=matching_text,
                    source=source,
                    extras={
                        "score": passage["score"],
                        "location": passage["location"],
                        "rlocation": passage["rlocation"],
                    },
                ),
            )

        return remove_duplicate_documents(documents)

    def _parse_sql_results(self, rows: List) -> List[Document]:
        """ "
        This method parses the string response from Sinequa into
        list of Document objects.

        Args:
            rows (List): list of results from SQL engine
        Returns:
            [Document] : list of Document objects
        """
        documents = []
        for row in rows:
            embeddings = ast.literal_eval(row[1])[0]["v"]

            documents.append(
                Document(
                    text=row[0],
                    embeddings=embeddings,
                    source=row[4],  # file_name
                ),
            )
        return documents

    def index_documents(self, paths: List[str]) -> Dict[str, BaseModel]:
        raise NotImplementedError

    def query(self, *args, **kwargs):
        raise NotImplementedError

    def _query_vectorstore(
        self,
        params: QueryParams,
        **kwargs,
    ) -> List[Document]:

        # search sinequa
        results = self.sinequa.search_query(params)
        return self._parse_query_results(results)

    def query_vectorstore(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        This method queries Sinequa to retrieve
        top_k documents with matching passages for given query text.

        Args:
            query (str): Query string
            top_k (int): Top k documents to surface
            collection (str): Collection to search
        Returns:
            List[Document]: Top k documents
        """

        # build query params
        params = QueryParams()
        params.search_text = query
        params.page = kwargs.get("page", 1)
        params.page_size = top_k * 2
        iterative_call = kwargs.get("iterative_call", False)

        # search for documents
        documents = self._query_vectorstore(params)

        # if iterative_call is True
        len_documents = len(documents)
        if len_documents < top_k and iterative_call:
            while len_documents < top_k:
                params.page += 1

                parsed_documents = self._query_vectorstore(params)
                if len(parsed_documents) == 0:
                    break

                documents.extend(parsed_documents)
                len_documents = len(documents)

        return documents[:top_k]

    def query_with_sql(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "user_needs_database",
        **kwargs,
    ) -> List[Document]:
        """
        This method queries Sinequa vector store to retrieve
        top_k documents with embeddings for given query text
        using direct SQL query.

        Args:
            query (str): Query string
            top_k (int): Top k documents to surface
            collection (str): Collection to search
        Returns:
            List[Document]: Top k documents
        """

        sql_query = self._generate_sql_query(query, collection, top_k)
        results = self.sinequa.engine_sql(
            sql=sql_query,
            max_rows=top_k,
        )
        return self._parse_sql_results(results["Rows"])
