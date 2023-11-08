#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from langchain.base_language import BaseLanguageModel
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import FAISS, VectorStore
from loguru import logger
from paperqa import Doc, Text
from pydantic import BaseModel
from pynequa import QueryParams, Sinequa
from tqdm import tqdm

from .metadata import AbstractMetadataExtractor, InstructorBasedOpenAIMetadataExtractor
from .paperqa_patched.docs import Docs
from .paperqa_patched.readers import read_doc_patched
from .processors import TextProcessor
from .structures import Document
from .utils import is_lambda, remove_duplicate_documents


class DocumentIndexer(ABC):
    """
    Abstract Base Class for setting up document indexer type
    """

    def __init__(
        self,
        docs: Optional[List[str]] = None,
        text_preprocessor: Optional[TextProcessor] = None,
        debug: bool = False,
    ) -> None:
        if is_lambda(text_preprocessor):
            raise TypeError(
                """Make sure text_preprocessor is not a lambda function.
                [Reason: can't pickle!]""",
            )

        self.text_preprocessor = text_preprocessor or (lambda x: x)
        self.debug = debug
        self._docs = docs or []
        self.doc_store = None

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
    def query(self, query: str, **kwargs) -> str:
        raise NotImplementedError()


class PaperQADocumentIndexer(DocumentIndexer):
    """
    Uses paperqa indexing pipeline to index documents provided that we have a list of paths.
    """

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        embeddings: Optional[Embeddings] = None,
        docs: Optional[List[str]] = None,
        doc_store: Optional[Docs] = None,
        text_preprocessor: Optional[Callable] = None,
        debug: bool = False,
        **paperqa_kwargs,
    ) -> None:
        super().__init__(docs=docs, text_preprocessor=text_preprocessor, debug=debug)

        llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        embeddings = embeddings or OpenAIEmbeddings(client=None)

        # if doc_store is provided externally, just use that
        # else construct a new one
        self.doc_store = doc_store or Docs(
            llm=llm,
            embeddings=embeddings,
            text_preprocessor=self.text_preprocessor,
            **paperqa_kwargs,
        )

    def index_documents(self, paths: List[str], **kwargs) -> PaperQADocumentIndexer:
        save_path = kwargs.get("save_path", None)
        _texts_len_original = len(self.texts)
        _docs = []
        for path in tqdm(paths):
            if path in self.docs:
                continue
            if self.debug:
                logger.debug(f"Creating index for src={path}")
            self.doc_store.add(path)
            _docs.append(path)
            if save_path is not None:
                self.save_index(save_path)
        self.docs.extend(_docs)

        if self.debug:
            _n_added = len(self.texts) - _texts_len_original
            logger.debug(f"Total of {len(_docs)} docs and {_n_added} pages indexed.")

        return self

    def save_index(self, path: str) -> PaperQADocumentIndexer:
        if not self.doc_store:
            return
        dump_val = dict(docs=self.docs, doc_store=self.doc_store)
        logger.info(f"Saving document index to {path}")
        with open(path, "wb") as f:
            pickle.dump(dump_val, f)
        return self

    def load_index(self, path: str) -> PaperQADocumentIndexer:
        with open(path, "rb") as f:
            dump_val = pickle.load(f)
            self.docs, self.doc_store = dump_val["docs"], dump_val["doc_store"]
        return self

    def query_vectorstore(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Queries the document/vector store
        """

        docs = []
        vecstore = self.doc_store.doc_index or self.doc_store.texts_index
        if vecstore is not None:
            docs = vecstore.similarity_search(query, k=top_k)
            docs = list(map(lambda d: Document.from_langchain_document(d), docs))
        return docs

    def query(self, query: str, **kwargs) -> str:
        if self.text_preprocessor is not None:
            query = self.text_preprocessor(query).strip()
        return self.doc_store.query(query).answer if query else ""

    @property
    def texts(self) -> List[Text]:
        return self.doc_store.texts


class LangchainDocumentIndexer(DocumentIndexer):
    """
    Uses paperqa indexing pipeline to index documents provided that we have a list of paths.
    """

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        embeddings: Optional[Embeddings] = None,
        docs: Optional[List[str]] = None,
        vector_store: Optional[VectorStore] = None,
        text_preprocessor: Optional[Callable] = None,
        text_splitter: Optional[TextSplitter] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(docs=docs, text_preprocessor=text_preprocessor, debug=debug)

        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.embeddings = embeddings or OpenAIEmbeddings(client=None)
        self.vector_store = vector_store
        self.text_splitter = text_splitter

    @property
    def docs(self) -> List[str]:
        _paths = []
        if self.vector_store is not None:
            _paths = map(
                lambda x: x.metadata.get("source", None),
                self.vector_store.docstore._dict.values(),
            )
        _paths = filter(None, _paths)
        return list(set(_paths))

    def qa_chain(self, k: int = 15) -> Type[Chain]:
        return RetrievalQA(
            combine_documents_chain=load_qa_chain(self.llm, chain_type="stuff"),
            retriever=self.vector_store.as_retriever(k=k),
        )

    def doc_store(self):
        return self.vector_store.docstore

    def _get_documents(self, paths: List[str]):
        if self.debug:
            logger.debug(f"Loading and preprocessing...")
        docs = []
        for path in tqdm(paths):
            if path in self.docs:
                continue
            _docs = PyPDFLoader(path).load()
            for _doc in _docs:
                _doc.page_content = self.text_preprocessor(_doc.page_content)
            docs.extend(_docs)
            self.docs.append(path)
        if self.text_splitter is not None and isinstance(
            self.text_splitter,
            TextSplitter,
        ):
            docs = self.text_splitter.split_documents(docs)
        return docs

    def index_documents(self, paths: List[str], **kwargs) -> LangchainDocumentIndexer:
        docs = self._get_documents(paths)
        if self.debug:
            logger.debug("Indexing...")
        if len(docs) < 1:
            logger.warning(
                f"Skipping indexing and returning existing vector store. Either empty docs or no new docs found!",
            )
            return self.vector_store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)
        if self.debug:
            logger.debug(f"Indexed {len(docs)} documents from {len(paths)} files.")
        return self

    def query_vectorstore(
        self,
        query: str,
        top_k=15,
        **kwargs,
    ) -> List[Document]:
        lang_docs = self.vector_store.similarity_search(query, k=top_k)
        return list(
            map(
                lambda d: Document.from_langchain_document(d),
                lang_docs,
            ),
        )

    def query(self, query: str, top_k=15, **kwargs) -> str:
        query = query.strip()
        if not query:
            return ""
        inp = dict(query=query, question=query)
        if self.debug:
            logger.debug(f"Using k={top_k}")
        result = self.qa_chain(k=top_k)(inp)
        if self.debug:
            logger.debug(result)
        return result.get("result", "").strip()

    def save_index(
        self,
        store_dir: str = "tmp/",
        index_name: str = "index",
    ) -> LangchainDocumentIndexer:
        self.vector_store.save_local(store_dir, index_name)
        return self

    def load_index(
        self,
        vector_store_cls: Type[VectorStore],
        store_dir: str,
        embeddings: "Embeddings",
        index_name: str = "index",
    ) -> LangchainDocumentIndexer:
        self.vector_store = vector_store_cls.load_local(
            store_dir,
            embeddings,
            index_name,
        )
        return self


class DocumentMetadataIndexer(DocumentIndexer):
    """
    This uses metadata extractor to index the metadata from documents.
    For now, we store metadata into a dictionary as an index.
    """

    def __init__(
        self,
        schema: Type[BaseModel],
        *,
        text_preprocessor: Optional[Callable] = None,
        metadata_extractor: Type[AbstractMetadataExtractor] = None,
        skip_errors: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug, text_preprocessor=text_preprocessor)
        self.metadata_extractor = (
            metadata_extractor
            or InstructorBasedOpenAIMetadataExtractor(
                model="gpt-3.5-turbo-0613",
                schema=schema,
                debug=debug,
            )
        )
        self.metadata_store = {}
        self.errors = set()
        self.skip_errors = skip_errors

    @property
    def schema(self) -> Type[BaseModel]:
        return self.metadata_extractor.schema

    def index_documents(self, paths: List[str], **kwargs) -> Dict[str, BaseModel]:
        save_path = kwargs.get("save_path", None)
        if self.debug:
            logger.debug(f"save_path = {save_path}")
        mstore = {}
        for p in tqdm(paths):
            if p in self.metadata_store:
                mstore[p] = self.metadata_store[p]
                continue

            text = read_doc_patched(
                p,
                Doc(citation=p, dockey=p, docname=p),
                text_preprocessor=self.text_preprocessor,
            )
            text = map(lambda x: x.text, text)
            text = "\n".join(text)

            hsh = str(hash(text))
            if hsh in self.metadata_store:
                mstore[hsh] = self.metadata_store[hsh]
                continue

            if self.debug:
                logger.debug(f"Extracting metadata from {p}")

            try:
                mstore[p] = self.metadata_extractor(text)
            except Exception as e:
                if self.skip_errors:
                    self.errors.add(p)
                    logger.debug(f"Skipping {p} | {str(e)}")
                    continue
                else:
                    raise e
            self.metadata_store.update(mstore)
            if save_path is not None:
                self.save_index(save_path)

        self.metadata_store.update(mstore)
        return self

    def save_index(self, path: str) -> None:
        dump_val = {}
        for k, v in self.metadata_store.items():
            dump_val[k] = v.dict()
        with open(path, "w") as f:
            json.dump(dump_val, f)
        return self

    def load_index(self, path: str) -> None:
        with open(path, "r") as f:
            dump_val = json.load(f)
            for k, v in dump_val.items():
                self.metadata_store[k] = self.schema.model_construct(**v)
        return self

    @property
    def doc_store(self) -> dict:
        return self.metadata_store

    @doc_store.setter
    def doc_store(self, value):
        self.metadata_store = value

    def query(self, *args, **kwargs):
        raise NotImplementedError()


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
        text_preprocessor: Optional[Callable] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            docs=docs,
            text_preprocessor=text_preprocessor,
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


def main():
    pass


if __name__ == "__main__":
    main()
