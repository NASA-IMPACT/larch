#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from langchain.schema.document import Document as LangchainDocument
from pydantic import BaseModel, SerializeAsAny


class Document(BaseModel):
    text: str
    page: Optional[int] = None
    embeddings: Optional[List[float]] = None
    source: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None

    @classmethod
    def from_langchain_document(cls, document: LangchainDocument) -> Document:
        doc = Document(text=document.page_content)
        metadata = deepcopy(document.metadata or {})
        doc.page = metadata.pop("page", None)
        doc.source = metadata.pop("source", None)
        doc.extras = metadata
        return doc

    def as_langchain_document(self, ignore_extras: bool = False) -> LangchainDocument:
        extras = {}
        if not ignore_extras:
            extras = (self.extras or {}).copy()
            extras["page"] = self.page
            extras["source"] = self.source
            extras["embeddings"] = self.embeddings
        return LangchainDocument(page_content=self.text, metadata=extras)

    def __str__(self) -> str:
        return str(self.text)


class Response(Document):
    evidences: SerializeAsAny[Optional[List[Document]]] = None

    def as_langchain_document(self, ignore_extras: bool = False) -> LangchainDocument:
        document = super().as_langchain_document(ignore_extras=ignore_extras)
        document.metadata["evidences"] = self.evidences
        return document


def main():
    pass


if __name__ == "__main__":
    main()
