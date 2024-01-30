#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from typing import Callable, Dict, List, Optional, Type

from langchain.base_language import BaseLanguageModel
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import FAISS, PGVector, VectorStore
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tqdm import tqdm

from ..metadata import AbstractMetadataExtractor, InstructorBasedOpenAIMetadataExtractor
from ..structures import Document, LangchainDocument, Response
from ..utils import remove_nulls
from ._base import DocumentIndexer


class LangchainDocumentIndexer(DocumentIndexer):
    """
    Uses custom langchain indexing pipeline to index documents provided that we have a list of paths.

    if provided `vector_store_params` is empty, a new default `vector_store` is initialized
    while calling `LangchainDocumentIndexer.load_index(...)`.

    The default class for the vector store is controlled by
    `LangchainDocumentIndexer.vector_store_params["default_vector_store_cls"]`.

    Usages:
        .. code-block: python

            from larch.indexing import LangchainDocumentIndexer

            # 1. directly using VectorStore object
            document_indexer = LangchainDocumentIndexer(
                llm=ChatOpenAI(model=model, temperature=0.0),
                text_processor=text_processor,
                vector_store=<vector_store>,
                # vector_store=FAISS.load_local("../tmp/vectorstore", embeddings=embedder, index_name="mistral_index"),
                debug=True,
            )

            # 2. Using params
            document_indexer = LangchainDocumentIndexer(
                llm=ChatOpenAI(model=model, temperature=0.0),
                text_processor=text_processor,
                vector_store_params=dict(pg_collection_name="vector_test", pg_connection_string="postgresql://postgresql@localhost:5432/vector_test"),
                #vector_store_params=dict(store_dir="../tmp/vectorstore/", index_name="faiss_test"),
                debug=True,
            )

    """

    _DEFAULT_VECTOR_STORE_CLS = FAISS
    _DEFAULT_PG_DISTANCE_STRATEGY = "cosine"

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        embeddings: Optional[Embeddings] = None,
        docs: Optional[List[str]] = None,
        vector_store: Optional[VectorStore] = None,
        text_processor: Optional[Callable] = None,
        text_splitter: Optional[TextSplitter] = None,
        vector_store_params: Optional[Dict] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(docs=docs, text_processor=text_processor, debug=debug)

        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.embeddings = embeddings or OpenAIEmbeddings(client=None)
        self.vector_store = vector_store
        self.text_splitter = text_splitter
        self.vector_store_params = vector_store_params or {}

        # create/build
        # If this still gives you None store,
        # A default new object is created at `load_index` time controleld by
        # `vector_store_params["default_vector_store_cls"]` to invoke its
        # `from_documents(...)` classmethod.
        self.vector_store = self._get_vector_store(self.vector_store_params)

    @property
    def docs(self) -> List[str]:
        _paths = []
        try:
            if self.vector_store is not None:
                _paths = map(lambda x: x.source, self.doc_store.values())
        except AttributeError:
            _paths = self._docs
        _paths = filter(None, _paths)
        return list(set(_paths))

    def _get_vector_store(self, params) -> Type[VectorStore]:
        """
        Obfuscated code that basically initializes the vetor store
        based on params if the provided vector_store object is None.
        """
        if self.vector_store is not None:
            return self.vector_store
        # check for non-pg first
        if "store_dir" in params and "index_name" in params:
            return params.get("vector_store_cls", FAISS).load_local(
                params["store_dir"],
                self.embeddings,
                params["index_name"],
            )
        if "pg_collection_name" in params and "pg_connection_string" in params:
            return PGVector(
                collection_name=params["pg_collection_name"],
                connection_string=params["pg_connection_string"],
                distance_strategy=params.get(
                    "pg_distance_strategy",
                    LangchainDocumentIndexer._DEFAULT_PG_DISTANCE_STRATEGY,
                ),
                embedding_function=self.embeddings,
            )

    def qa_chain(self, k: int = 15, **kwargs) -> Type[Chain]:
        search_params = kwargs.copy()
        search_params["filter"] = search_params.pop("filter_by", {})
        return RetrievalQA(
            combine_documents_chain=load_qa_chain(
                self.llm,
                chain_type=kwargs.get("chain_type", "stuff"),
            ),
            retriever=self.vector_store.as_retriever(k=k, search_kwargs=search_params),
            return_source_documents=True,
        )

    def _get_pgvector_doc_store(self) -> Dict[str, Document]:
        table_name = self.vector_store.EmbeddingStore.__tablename__
        sql_query = f"SELECT uuid, document, cmetadata from {table_name};"

        # filter by collection id
        with Session(self.vector_store._conn) as session:
            collection = self.vector_store.get_collection(session)
            if collection is not None:
                sql_query = (
                    f"SELECT uuid, document, cmetadata from {table_name}"
                    + f" where collection_id='{collection.uuid}';"
                )

        # create the dictionary store
        store = {}
        for row in self.vector_store._conn.exec_driver_sql(sql_query):
            uuid = row[0]
            document = row[1]
            metadata = row[2]
            store[uuid] = Document(
                text=document,
                page=metadata.get("page"),
                source=metadata.get("source"),
            )
        return store

    def _get_faiss_doc_store(self) -> Dict[str, Document]:
        if self.vector_store is None:
            return {}
        store = {}
        for uuid, doc in self.vector_store.docstore._dict.items():
            store[uuid] = Document.from_langchain_document(doc)
        return store

    @property
    def doc_store(self) -> Dict[str, Document]:
        """
        This generates a consistent dict structure
        no matter what vector store is used.
        Key is uuid of the chunk and the value is the
        chunk information (object initialized at `larch.structures.Document`)
        """
        store = {}
        if isinstance(self.vector_store, FAISS):
            store = self._get_faiss_doc_store()
        elif isinstance(self.vector_store, PGVector):
            store = self._get_pgvector_doc_store()
        return store

    @property
    def num_chunks(self) -> int:
        """
        Returns total number of chunks in the vector store.
        """
        return len(self.doc_store)

    def index_documents(self, paths: List[str], **kwargs) -> LangchainDocumentIndexer:
        store_dir = kwargs.pop("store_dir", None)
        index_name = kwargs.pop("index_name", None)

        paths = self._get_new_paths(paths)
        doc_map = self._get_documents(paths, text_splitter=self.text_splitter, **kwargs)
        docs = list(itertools.chain(*doc_map.values()))

        self.docs.extend(paths)
        if len(docs) < 1:
            logger.warning(
                "Skipping indexing. Either empty docs or no new docs found!",
            )
            return self

        # build index
        _prev_chunks_num = self.num_chunks
        if self.vector_store is None:
            vs_cls = self.vector_store_params.get(
                "default_vector_store_cls",
                self._DEFAULT_VECTOR_STORE_CLS,
            )
            self.vector_store = vs_cls.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)

        if self.debug:
            logger.debug(
                f"Indexed {self.num_chunks-_prev_chunks_num} chunks from {len(paths)} files.",
            )

        if store_dir and index_name:
            self.save_index(store_dir=store_dir, index_name=index_name)
        return self

    def query_vectorstore(
        self,
        query: str,
        top_k=15,
        **kwargs,
    ) -> List[Document]:
        if self.vector_store is None:
            logger.warning("vector_store not initialized!")
            return []
        kwargs = kwargs.copy()
        filter_by = remove_nulls(kwargs.pop("filter_by", {}))
        if self.debug:
            logger.debug(f"top_k={top_k}")
            logger.debug(f"filter_by = {filter_by}")

        chunks = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k,
            filter=filter_by,
            **kwargs,
        )
        results = []
        # tuple of (doc, score)
        for chunk in chunks:
            doc = Document.from_langchain_document(chunk[0])
            doc.extras["score"] = chunk[1]
            results.append(doc)
        return results

    def query(self, query: str, top_k=5, **kwargs) -> Response:
        query = query.strip()
        if not query:
            return Response(text="")
        inp = dict(query=query, question=query)
        if self.debug:
            logger.debug(f"Using k={top_k}")
        result = self.qa_chain(k=top_k, **kwargs)(inp)
        if self.debug:
            logger.debug(result)
        return Response(
            text=result.get("result", "").strip(),
            evidences=list(
                map(
                    Response.from_langchain_document,
                    result.get("source_documents", []),
                ),
            ),
        )

    def save_index(
        self,
        store_dir: str = "tmp/",
        index_name: str = "index",
    ) -> LangchainDocumentIndexer:
        if not isinstance(self.vector_store, PGVector):
            self.vector_store.save_local(store_dir, index_name)
        return self

    def load_index(
        self,
        vector_store_cls: Type[VectorStore],
        store_dir: str,
        embeddings: "Embeddings",
        index_name: str = "index",
    ) -> LangchainDocumentIndexer:
        if not isinstance(vector_store_cls, PGVector):
            self.vector_store = self._get_vector_store(
                dict(
                    store_dir=store_dir,
                    index_name=index_name,
                    vector_store_cls=vector_store_cls,
                ),
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
        text_processor: Optional[Callable] = None,
        metadata_extractor: Type[AbstractMetadataExtractor] = None,
        skip_errors: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug, text_processor=text_processor)
        self.metadata_extractor = (
            metadata_extractor
            or InstructorBasedOpenAIMetadataExtractor(
                model="gpt-3.5-turbo-0613",
                schema=schema,
                debug=debug,
            )
        )
        self.metadata_store = {}
        self._doc_store = {}
        self.errors = set()
        self.skip_errors = skip_errors

    @property
    def schema(self) -> Type[BaseModel]:
        return self.metadata_extractor.schema

    def index_documents(
        self,
        paths: List[str],
        chunk_separator: str = "\n",
        **kwargs,
    ) -> DocumentMetadataIndexer:
        # to double check which argument has the actual path to save metadata
        save_path = kwargs.get("save_path", kwargs.get("metadata_path", None))
        doc_store_path = kwargs.get("doc_store_path", None)
        if self.debug:
            logger.debug(f"save_path = {save_path} | doc_store_path={doc_store_path}")

        paths = self._get_new_paths(paths)
        doc_map = self._get_documents(paths, text_splitter=None)
        chunk_separator = str(chunk_separator)
        for path, docs in tqdm(doc_map.items()):
            # skip if already present
            if path in self.metadata_store:
                continue

            self._doc_store[path] = docs
            text = chunk_separator.join(map(lambda x: x.page_content, docs))

            if self.debug:
                logger.debug(f"Extracting metadata from {path}")

            try:
                self.metadata_store[path] = self.metadata_extractor(text)
            except Exception as e:
                if self.skip_errors:
                    self.errors.add(path)
                    logger.debug(f"Skipping {path} | {str(e)}")
                    continue
                else:
                    raise e
            if save_path is not None:
                self.save_index(save_path, doc_store_path=doc_store_path)
        return self

    def _dump_doc_store(self, dump_path: str) -> bool:
        doc_store = {}
        if dump_path is None:
            return False
        for path, docs in self.doc_store.items():
            doc_store[path] = list(map(lambda doc: doc.dict(), docs))
        if not dump_path.endswith(".json"):
            dump_path = f"{dump_path}.json"
        with open(dump_path, "w") as f:
            json.dump(doc_store, f)
        return True

    def _dump_metadata_store(self, dump_path: str) -> bool:
        if dump_path is None:
            return False
        dump_val = {}
        for path, vals in self.metadata_store.items():
            dump_val[path] = vals.dict()
        with open(dump_path, "w") as f:
            json.dump(dump_val, f)
        return True

    def save_index(
        self,
        metadata_path: str,
        doc_store_path: Optional[str] = None,
    ) -> None:
        """
        Saves extracted metadata as well as the document parsed.
        Args:
            ```metadata_path```: ```str```
                json path to save all the metadata extracted.
                It saves `DocumentMetadataIndexer.metadata_store` dict
            ```doc_store_path```: ```Optional[str]```
                If provided, this is the json file to which
                the extracted documents (full) are saved to.
                It saves `DocumentMetadataIndexer.doc_store` object.
        """
        self._dump_metadata_store(metadata_path)
        self._dump_doc_store(doc_store_path)
        return self

    def _load_doc_store(
        self,
        doc_store_path: str,
    ) -> Dict[str, List[LangchainDocument]]:
        if doc_store_path is None:
            return {}
        doc_store = {}
        with open(doc_store_path, "r") as f:
            dump_val = json.load(f)
            for path, docs in dump_val.items():
                doc_store[path] = list(
                    map(lambda d: LangchainDocument.construct(**d), docs),
                )
        return doc_store

    def _load_metadata_store(self, metadata_path: str) -> Dict[str, BaseModel]:
        if metadata_path is None:
            return {}
        metadata_store = {}
        with open(metadata_path, "r") as f:
            dump_val = json.load(f)
            for k, v in dump_val.items():
                metadata_store[k] = self.schema.model_construct(**v)
        return metadata_store

    @property
    def docs(self) -> List[str]:
        """
        Returns a list of files/paths that are already extracted.
        """
        return list(self.metadata_store.keys())

    def load_index(
        self,
        metadata_path: str,
        doc_store_path: Optional[str] = None,
    ) -> None:
        """
        Loads extracted metadata as well as the full documents.
        """
        self.metadata_store.update(self._load_metadata_store(metadata_path))
        self._doc_store.update(self._load_doc_store(doc_store_path))
        return self

    def query(self, *args, **kwargs):
        raise NotImplementedError()
