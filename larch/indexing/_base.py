#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union

from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import TextSplitter
from loguru import logger
from tqdm import tqdm

from ..processors import TextProcessor
from ..structures import Document, LangchainDocument, Response
from ..utils import LangchainDocumentParser, is_lambda


class DocumentIndexer(ABC):
    """
    Abstract Base Class for setting up document indexer type
    """

    def __init__(
        self,
        docs: Optional[List[str]] = None,
        text_processor: Optional[Union[Callable, TextProcessor]] = None,
        debug: bool = False,
    ) -> None:
        if is_lambda(text_processor):
            raise TypeError(
                """Make sure text_processor is not a lambda function.
                [Reason: can't pickle!]""",
            )

        self.text_processor = text_processor or (lambda x: x)
        self.debug = debug
        # list of paths added
        self._docs = docs or []
        self._doc_store = None

    def _get_documents(
        self,
        paths: List[str],
        text_splitter: Optional[TextSplitter] = None,
        **metadata,
    ) -> Dict[str, List[LangchainDocument]]:
        """
        This parses all the files and returns a dictionary mapping
        from file path to langchain document objects.
        """
        if self.debug:
            logger.debug("Loading...")

        # if not provided externally, get from the object, else just None
        text_splitter = text_splitter or getattr(self, "text_splitter", None)
        doc_parser = LangchainDocumentParser(text_splitter=text_splitter)
        doc_map = doc_parser(paths)

        # pre-process texts
        if self.debug:
            logger.debug("Preprocessing texts...")
        for path, docs in tqdm(doc_map.items()):
            for _doc in docs:
                _doc.page_content = self.text_processor(_doc.page_content)
                _doc.metadata.update(**metadata)
        return doc_map

    def _get_new_paths(self, paths: List[str]) -> List[str]:
        """
        This filters the paths to only get the files that aren't
        indexed yet.
        """
        if isinstance(paths, str):
            paths = [paths]
        docs = self.docs
        return list(filter(lambda path: path not in docs, paths))

    @property
    def doc_store(self):
        return self._doc_store

    @doc_store.setter
    def doc_store(self, value):
        self._doc_store = value

    @property
    def docs(self) -> List[str]:
        return self._docs

    @docs.setter
    def docs(self, x):
        self._docs = x

    @abstractmethod
    def index_documents(self, docs: List[str], **kwargs) -> Any:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Any:
        return self.index_documents(*args, **kwargs)

    def save_index(self, path: str):
        raise NotImplementedError()

    def load_index(self, path: str):
        raise NotImplementedError()

    def query_vectorstore(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """
        Queries the document/vector store
        """
        raise NotImplementedError()

    def query_top_k(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        return self.query_vectorstore(query, top_k, **kwargs)

    @abstractmethod
    def query(self, query: str, **kwargs) -> Response:
        raise NotImplementedError()

    def as_langchain_retriever(self, top_k: int = 5) -> Type[BaseRetriever]:
        # to avoid circular import
        from ..search.chains import DocumentIndexerAsRetriever

        return DocumentIndexerAsRetriever(document_indexer=self, top_k=top_k)
