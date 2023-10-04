#!/usr/bin/env python3

import json
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from loguru import logger
from paperqa import Doc, Text
from pydantic import BaseModel
from tqdm import tqdm

from .metadata import AbstractMetadataExtractor, InstructorBasedOpenAIMetadataExtractor
from .paperqa_patched.docs import Docs
from .paperqa_patched.readers import read_doc_patched
from .utils import is_lambda


class DocumentIndexer(ABC):
    """
    Abstract Base Class for setting up document indexer type
    """

    def __init__(
        self,
        docs: Optional[List[str]] = None,
        text_preprocessor: Optional[Callable] = None,
        debug: bool = False,
    ) -> None:
        if is_lambda(text_preprocessor):
            raise TypeError(
                "Make sure text_preprocessor is not a lambda function. [Reason: can't pickle!]",
            )

        self.text_preprocessor = text_preprocessor or (lambda x: x)
        self.debug = debug
        self.docs = docs or []
        self.doc_store = None

    @abstractmethod
    def index_documents(self, docs: List[str], **kwargs) -> Any:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Any:
        return self.index_documents(*args, **kwargs)

    def save_index(self, path: str):
        raise NotImplementedError()

    def load_index(self, path: str):
        raise NotImplementedError()

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

    def index_documents(self, paths: List[str], **kwargs) -> Any:
        for path in tqdm(paths):
            if path in self.docs:
                continue
            if self.debug:
                logger.debug(f"Creating index for src={path}")
            self.doc_store.add(path)
        if self.debug:
            logger.debug(f"Total of {len(self.doc_store.texts)} pages indexed.")

        self.docs.extend(paths)
        return self

    def save_index(self, path: str):
        if not self.doc_store:
            return
        dump_val = dict(docs=self.docs, doc_store=self.doc_store)
        logger.info(f"Saving document index to {path}")
        with open(path, "wb") as f:
            pickle.dump(dump_val, f)
        return self

    def load_index(self, path: str):
        with open(path, "rb") as f:
            dump_val = pickle.load(f)
            self.docs, self.doc_store = dump_val["docs"], dump_val["doc_store"]
        return self

    def query(self, query: str, **kwargs) -> str:
        return self.doc_store.query(query).answer

    @property
    def texts(self) -> List[Text]:
        return self.doc_store.texts


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

    @property
    def schema(self) -> Type[BaseModel]:
        return self.metadata_extractor.schema

    def index_documents(self, paths: List[str]) -> Dict[str, BaseModel]:
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

            mstore[p] = self.metadata_extractor(text)

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


def main():
    pass


if __name__ == "__main__":
    main()
