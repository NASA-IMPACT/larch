from typing import Any, Dict, List, Type

from langchain.text_splitter import TokenTextSplitter
from loguru import logger
from pydantic import BaseModel

from ..utils import remove_nulls
from ._base import MetadataAggregator
from .extractors_openai import InstructorBasedOpenAIMetadataExtractor


class TokenChunker:
    def __init__(self, chunk_size: int = 2048, chunk_overlap: int = 256) -> None:
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        spliter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return spliter.split_text(text)

    def __call__(self, *args, **kwargs) -> List[str]:
        return self.split_text(*args, **kwargs)


class InstructorAggregator(MetadataAggregator):
    _aggregation_prompt: str = (
        "Take this list of dictionaries, and consolidate it into a single extraction"
    )

    def __init__(
        self,
        schema: Type[BaseModel],
        model: str = "gpt-3.5-turbo-0613",
        aggregation_prompt: str = _aggregation_prompt,
        remove_nulls: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(remove_nulls=remove_nulls, debug=debug)
        self.schema: Type[BaseModel] = schema
        self._aggregation_prompt: str = aggregation_prompt
        self.extractor = InstructorBasedOpenAIMetadataExtractor(
            model=model,
            schema=self.schema,
            system_prompt=self._aggregation_prompt,
            debug=debug,
        )

    def _serialize_chunks(self, extractions: List[Type[BaseModel]]) -> str:
        """Converts a list of pydantic metadata extractions into a string of dictionaries to
        be passed to the aggregating extractor"""

        extraction_dicts = [e.model_dump() for e in extractions]
        if self.remove_nulls:
            extraction_dicts = list(map(remove_nulls, extraction_dicts))

        if self.debug:
            logger.debug(f"Serialized chunks :: \n{extraction_dicts}")

        return "\n\n".join(map(str, extraction_dicts))

    def aggregate(self, extractions: List[Dict[str, Any]]) -> Type[BaseModel]:
        if self.debug:
            logger.debug(f"Running aggregations on {len(extractions)} extractions.")
        return self.extractor(self._serialize_chunks(extractions))

    def __call__(self, *args, **kwargs) -> Type[BaseModel]:
        return self.aggregate(*args, **kwargs)
