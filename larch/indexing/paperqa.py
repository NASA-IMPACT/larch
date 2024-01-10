#!/usr/bin/env python3
from __future__ import annotations

import pickle
from typing import Callable, List, Optional, Type

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import VectorStore
from loguru import logger
from paperqa import Text
from tqdm import tqdm

from ..utils import remove_duplicate_documents, remove_nulls
from ._base import DocumentIndexer
from .paperqa_patched.docs import Docs
from .structures import Document, Response


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
        text_processor: Optional[Callable] = None,
        debug: bool = False,
        **paperqa_kwargs,
    ) -> None:
        super().__init__(docs=docs, text_processor=text_processor, debug=debug)

        llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        embeddings = embeddings or OpenAIEmbeddings(client=None)

        # if doc_store is provided externally, just use that
        # else construct a new one
        self._doc_store = doc_store or Docs(
            llm=llm,
            embeddings=embeddings,
            text_processor=self.text_processor,
            **paperqa_kwargs,
        )

    def index_documents(self, paths: List[str], **kwargs) -> PaperQADocumentIndexer:
        save_path = kwargs.get("save_path", None)
        doc_type = kwargs.get("doc_type", None)
        if isinstance(paths, str):
            paths = [paths]

        prevous_n_texts = len(self.texts)

        paths = self._get_new_paths(paths)
        for path in tqdm(paths):
            if path in self.docs:
                continue
            if self.debug:
                logger.debug(f"Creating index for src={path}")
            self.doc_store.add(path, doc_type=doc_type)
            self.docs.append(path)
            if save_path:
                self.save_index(save_path)
        self.doc_store._build_texts_index()
        self.save_index(save_path)

        if self.debug:
            logger.debug(
                f"Total of {len(paths)} docs and {len(self.texts) - prevous_n_texts} chunks indexed.",
            )

        return self

    def save_index(self, path: str) -> PaperQADocumentIndexer:
        if not self.doc_store:
            return self
        if not path:
            return self

        # hack to prevent error due to lambda functions :/
        if path is not None:
            self.doc_store.text_processor = None

        dump_val = dict(docs=self.docs, doc_store=self.doc_store)
        logger.info(f"Saving document index to {path}")
        with open(path, "wb") as f:
            pickle.dump(dump_val, f)
        self.doc_store.text_processor = self.text_processor
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
        vecstore = self.doc_store.texts_index
        if vecstore is None:
            return []

        filter_by = remove_nulls(kwargs.get("filter_by", {}))
        if self.debug:
            logger.debug(f"filter_by = {filter_by}")
        docs = vecstore.similarity_search(
            query,
            k=top_k,
            filter=filter_by,
        )
        for doc in docs:
            if self.text_processor is not None:
                doc.page_content = self.text_processor(doc.page_content)

        # Convert to larch format and remove any duplicates resulting from
        # multiple filters
        docs = list(map(lambda d: Document.from_langchain_document(d), docs))
        docs = remove_duplicate_documents(docs)

        # update source
        for doc in docs:
            doc.source = doc.extras.get("doc", {}).get("citation", None)
        return docs

    def query(self, query: str, top_k: int = 10, **kwargs) -> Response:
        if self.text_processor is not None:
            query = self.text_processor(query).strip()
        pqa_res = self.doc_store.query(query, k=top_k, max_sources=top_k, **kwargs)
        return Response(
            text=pqa_res.answer,
            evidences=[
                Document(text=context.text.text, source=context.text.doc.citation)
                for context in pqa_res.contexts
            ],
        )

    @property
    def texts(self) -> List[Text]:
        return self.doc_store.texts

    @property
    def vector_store(self) -> Type[VectorStore]:
        return self.doc_store.texts_index


def main():
    pass


if __name__ == "__main__":
    main()
