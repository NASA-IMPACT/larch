#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.schema.document import Document as LangchainDocument


@dataclass
class Document:
    text: str
    page: Optional[int] = None
    embeddings: Optional[List[float]] = None
    source: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None

    @classmethod
    def from_langchain_document(cls, document: LangchainDocument) -> Document:
        doc = Document(text=document.page_content)
        metadata = document.metadata or {}
        doc.page = metadata.get("page", None)
        doc.source = metadata.get("source", None)
        return doc


def main():
    pass


if __name__ == "__main__":
    main()
