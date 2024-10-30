#!/usr/bin/env python3

import ast
import json
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pynequa import AdvancedParams, QueryParams, Sinequa

from ..structures import Document
from ..utils import remove_duplicate_documents
from ._base import DocumentRetriever


class SinequaDocumentRetriever(DocumentRetriever):
    """
    This retriever uses Sinequa as document store to retriever top passages based on query.

    This uses `search.query` endpoint.
    """

    # not recommended to change as it might break result parsing
    # | For new columns, just append to the end of the list to prevent
    # change of indexing issues during parsing. |
    _columns_to_surface = [
        "text",
        "passagevectors",
        "collection",
        "treepath",
        "filename",
        "GlobalRelevance",
        "id",
    ]

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        app_name: str = "vanilla-search",
        query_name: str = "query",
        index: Optional[str] = None,
        collection: Optional[str] = None,
        columns: Optional[List[str]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
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
        self.columns = columns or self._columns_to_surface
        self.index = index
        self.collection = collection

    @property
    def column_to_idx(self) -> Dict[str, int]:
        return {col: idx for idx, col in enumerate(self.columns)}

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
                        "record_id": passage["recordId"],
                    },
                ),
            )

        return remove_duplicate_documents(documents)

    def _query_search(
        self,
        params: QueryParams,
        **kwargs,
    ) -> List[Document]:
        # if self.debug:
        #     logger.debug(f"pynequa params :: {params}")
        #     logger.debug(f"Generated Payload :: {params.generate_payload()}")
        results = self.sinequa.search_query(params)
        return self._parse_query_results(results)

    def _build_advanced_params(
        self,
        **kwargs,
    ) -> Union[AdvancedParams, List[AdvancedParams]]:
        collection = self.collection
        advanced_params = kwargs.get("advanced") or kwargs.get("advanced_params") or []

        # if no params supplied externally, build one for collection name if
        # possible
        if collection and not advanced_params:
            advanced_params = AdvancedParams(
                col_name="collection",
                col_value=collection,
            )
        # if list is supplied, and each item is a dict
        elif (
            advanced_params
            and isinstance(advanced_params, list)
            and isinstance(advanced_params[0], dict)
        ):
            advanced_params = list(map(lambda p: AdvancedParams(**p), advanced_params))
        # if single dict, just build it
        elif advanced_params and isinstance(advanced_params, dict):
            advanced_params = AdvancedParams(**advanced_params)
        return advanced_params

    def _build_pynequa_params(self, **kwargs) -> QueryParams:
        params = QueryParams()
        params.search_text = kwargs.get("query")
        params.page = kwargs.get("page", 1)
        params.page_size = kwargs.get("top_k", 5) * 2
        params.additional_where_clause = kwargs.get("additional_where_clause")
        params.advanced = self._build_advanced_params(**kwargs)
        params.debug = kwargs.get("debug") or self.debug
        return params

    def query_top_k(
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
            kwargs: Consist of different params to call Sinequa's query API
        Returns:
            List[Document]: Top k documents
        """
        iterative_call = kwargs.get("iterative_call", False)
        params: QueryParams = self._build_pynequa_params(
            query=query,
            top_k=top_k,
            **kwargs,
        )

        documents: List[Document] = self._query_search(params)

        len_documents = len(documents)
        if len_documents < top_k and iterative_call:
            while len_documents < top_k:
                params.page += 1

                parsed_documents = self._query_search(params)
                if len(parsed_documents) == 0:
                    break

                documents.extend(parsed_documents)
                len_documents = len(documents)

        return documents[:top_k]


class SinequaSQLRetriever(SinequaDocumentRetriever):
    """
    A Retriever type for implementing SQL-based retrival.
    It uses Sinequa's SQL engine.
    """

    _sql = """SELECT {columns} FROM {index}
    WHERE collection='{collection}'
    AND text contains '{query}'
    AND SearchParameters='{search_params}'
    LIMIT {limit}
    """.strip()

    _data_access_key = "Rows"

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        app_name: str = "vanilla-search",
        query_name: str = "query",
        index: Optional[str] = None,
        collection: Optional[str] = None,
        columns: Optional[List[str]] = None,
        search_parameters: Optional[Dict] = None,
        sql: Optional[str] = None,
        neural_search: bool = True,
        text_column_name="text",
        debug: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url,
            auth_token=auth_token,
            app_name=app_name,
            query_name=query_name,
            index=index,
            collection=collection,
            columns=columns,
            debug=debug,
        )
        self.sql = sql or self._sql
        self.neural_search = bool(neural_search)
        self.search_parameters = search_parameters or {}
        self.text_column_name = text_column_name
        # adding text_column_name to columns list if it doesn't exist already
        self.columns = self.columns + [self.text_column_name] if text_column_name not in self.columns else self.columns

    def query_top_k(
        self,
        query: str,
        top_k: int = 5,
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

        collection = (
            self.collection or kwargs.get("collection") or kwargs.get("collection_name")
        )
        sql_query = self._generate_sql_query(
            query,
            collection=collection,
            index=self.index,
            limit=top_k,
        )
        if self.debug:
            logger.debug(f"SQL Query :: {sql_query}")
        results = self.sinequa.engine_sql(
            sql=sql_query,
            max_rows=top_k,
        )
        self._validate_result(results)
        return self._parse_sql_results(results[self._data_access_key])

    def _validate_result(self, result: dict) -> bool:
        """
        Validate if error or not.
        """
        methodresult = result.get("methodresult", "").lower()
        if "error" in methodresult:
            raise ValueError(f"Unable to get result | Error => {result}")
        if self._data_access_key not in result:
            raise ValueError(
                f"Unable to get result | '{self._data_access_key}' not present | Result => {result}",
            )

        return True

    @property
    def _search_params_str(self) -> str:
        if not self.search_parameters:
            return ""
        res = ""
        for k, v in self.search_parameters.items():
            res += f"{k}={v};"
        return res

    def _generate_sql_query(
        self,
        query: str,
        collection: str,
        index: str,
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
        column_str = ",".join(self.columns)
        limit = limit if limit else -1
        query = query.replace("'", "''")
        sql = self.sql.format(
            columns=column_str,
            index=self.index,
            collection=collection,
            query=query,
            search_params=self._search_params_str,
            limit=limit,
        )
        return sql

    def _parse_sql_results(self, rows: List) -> List[Document]:
        raise NotImplementedError()


class SinequaSQLDocumentRetriever(SinequaSQLRetriever):
    """
    This retriever uses Sinequa as document store to retriever top documents
    (full text, not passages) based on query.
    It uses Sinequa's SQL engine to get the relevant docs.

    """

    def _parse_passagevectors(self, row: dict) -> Optional[List[float]]:
        """
        Helper to parse passage vector
        """
        key = list(filter(lambda x: "passagevector" in x.lower(), self.columns))
        if not key:
            return None
        vec = row.pop(key[0])
        vec = list(map(lambda x: x.get("v", None), ast.literal_eval(vec)))
        vec = list(filter(None, vec)) or None
        return np.mean(vec, axis=0).tolist() if vec else None

    def _parse_sql_results(self, rows: List) -> List[Document]:
        """Parses SQL result rows into a list of Document objects, adapting to available columns.

        Args:
            rows (List): List of results from SQL engine.

        Returns:
            List[Document]: List of Document objects.
        """
        documents = []
        column_map = self.column_to_idx
        for row in rows:
            dct = dict(zip(self.columns, row))
            documents.append(
                Document(
                    text=dct.pop(self.text_column_name),
                    embeddings=self._parse_passagevectors(dct),
                    source=dct.pop("filename"),
                    extras={
                        **{"score": dct.pop("GlobalRelevance")},
                        **dct,
                    },
                ),
            )

        return documents


class SinequaSQLPassageRetriever(SinequaSQLRetriever):
    """
    Retrieve passages/chunks.
    For retrieving full document, use `SinequaSQLRetriever`
    """

    _data_access_key = "Attributes"

    _sql = """
select
    TOPPASSAGES('columns=id/filename, matchlocations=(),count={limit},minscore=0') as TP
from
   {index}
where
    collection='{collection}'
    and text contains '{query}'
    and SearchParameters='{search_params}'
    limit {limit};
    """

    def _parse_sql_results(self, rows: dict) -> List[Document]:
        """ "
        This method parses the string response from Sinequa into
        list of Document objects.

        Args:
            rows (List): list of results from SQL engine
        Returns:
            [Document] : list of Document objects
        """
        res = []
        passages = json.loads(rows.get("tp", "[]")).get("passages", [])
        for passage in passages:
            columns = {k: v for c in passage.pop("columns", []) for k, v in c.items()}
            res.append(
                Document(
                    text=passage.pop(self.text_column_name, None),
                    source=columns.pop("id", None),
                    extras={**passage, **columns},
                ),
            )
        return res
