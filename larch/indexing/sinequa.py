#!/usr/bin/env python3
import ast
from typing import Callable, Dict, List, Optional
from warnings import warn

from loguru import logger
from pydantic import BaseModel
from pynequa import AdvancedParams, QueryParams, Sinequa

from ..retrievers import SinequaDocumentRetriever
from ..structures import Document
from ..utils import remove_duplicate_documents
from ._base import DocumentIndexer

DEPRECATION_MESSAGE = "`larch.indexing.SinequaDocumentIndexer` will be deprecated in favor of `larch.retrievers.SinequaDocumentRetriever` for larch version`>=0.1.0`"


class SinequaDocumentIndexer(DocumentIndexer):
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
        collection: Optional[str] = None,
        columns: Optional[List[str]] = columns_to_surface,
        docs: Optional[List[str]] = None,
        text_processor: Optional[Callable] = None,
        debug: bool = False,
    ) -> None:
        logger.warning(DEPRECATION_MESSAGE)
        warn(
            DEPRECATION_MESSAGE,
            DeprecationWarning,
            2,
        )
        super().__init__(
            docs=docs,
            text_processor=text_processor,
            debug=debug,
        )

        self.retriever = SinequaDocumentRetriever(
            base_url=base_url,
            auth_token=auth_token,
            app_name=app_name,
            query_name=query_name,
            collection=collection,
            columns=columns,
            debug=debug,
        )

    def index_documents(self, paths: List[str]) -> Dict[str, BaseModel]:
        raise NotImplementedError

    def query(self, *args, **kwargs):
        raise NotImplementedError

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
        return self.retriever(query=query, top_k=top_k, **kwargs)
