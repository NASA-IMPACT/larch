#!/usr/bin/env python3
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from .._base import AbstractClass
from ..structures import Document, LangchainDocument


class DocumentRetriever(AbstractClass):
    """
    Base class to represent retriever within larch
    """

    @abstractmethod
    def query_top_k(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        raise NotImplementedError()

    def query(self, *args, **kwargs) -> List[Document]:
        return self.query_top_k(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> List[Document]:
        return self.query_top_k(*args, **kwargs)

    def as_langchain_retriever(self, **kwargs) -> Type[BaseRetriever]:
        top_k = kwargs.pop("top_k", 5)
        return LangchainDocumentRetriever(retriever=self, top_k=top_k, params=kwargs)


class LangchainDocumentRetriever(BaseRetriever):
    """
    This is a wrapper that makes larch's DocumentRetriever
    compatible with langchain's retriever
    """

    retriever: DocumentRetriever
    top_k: int = 5
    params: Optional[Dict[Any, Any]] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs,
    ) -> List[LangchainDocument]:
        params = {**kwargs, **(self.params or {})}
        documents = self.retriever.query_top_k(query, top_k=self.top_k, **params)
        return list(map(Document.as_langchain_document, documents))
