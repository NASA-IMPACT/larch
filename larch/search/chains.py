#!/usr/bin/env python3

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from ..indexing import DocumentIndexer
from ..structures import Document, LangchainDocument


class DocumentIndexerAsRetriever(BaseRetriever):
    document_indexer: DocumentIndexer
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs,
    ) -> List[LangchainDocument]:
        documents = self.document_indexer.query_top_k(query, top_k=self.top_k, **kwargs)
        return list(map(Document.as_langchain_document, documents))


def main():
    pass


if __name__ == "__main__":
    main()
