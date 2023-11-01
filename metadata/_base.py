#!/usr/bin/env python3

import re
from abc import ABC, abstractmethod
from functools import cache
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AbstractMetadataExtractor(ABC):
    """
    Each MetadataExtractor can be used like a functor through __call__.

    Any downstream implementation/subclass should implement `_extract(...)`
    method.

    Args:
        ```preprocessor```: ```Optional[Callable]```
            A callable that preprocessing input text string.
            Defaults to merging multiple whitespace into single.
        ```debug```: ```bool```
            Flag to enable debug mode logs.
            Defaults to `False`
    """

    def __init__(
        self,
        preprocessor: Optional[Callable] = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.preprocessor = preprocessor or self._preprocess_text

    def extract_metadata(self, text: str) -> Optional[Type[BaseModel]]:
        """
        Extract metadata from the input text

        Args:
            ```text```: ```str```
                Input text from which metadata is extracted

        Returns:
            Return a pydantic schema object to map the metadata
        """
        text = self.preprocessor(text)
        return self._extract(text) if text else None

    @cache
    @abstractmethod
    def _extract(self, text: str):
        raise NotImplementedError()

    @staticmethod
    def _preprocess_text(text: str):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def __call__(self, *args, **kwargs):
        return self.extract_metadata(*args, **kwargs)


class MetadataValidator(ABC):
    """
    A component to validate values from the extracted metadata.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @abstractmethod
    def validate(self, metadata: Type[BaseModel], **kwargs) -> Type[BaseModel]:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Type[BaseModel]:
        """
        Performs validation for the provided metadata.
        """
        return self.validate(*args, **kwargs)


class MetadataEvaluator(ABC):
    """
    A component to evaluate extracted metadata with provided reference
    """

    def __init__(self, ignore_case: bool = True, debug: bool = False) -> None:
        self.debug = debug
        self.ignore_case = ignore_case

    @abstractmethod
    def evaluate(
        self,
        prediction: Union[Type[BaseModel], Dict],
        reference: Union[Type[BaseModel], Dict],
        **kwargs,
    ) -> Any:
        raise NotImplementedError()

    def _get_dict(self, metadata: Any) -> dict:
        if isinstance(metadata, BaseModel):
            metadata = metadata.model_dump()
        return metadata

    def __call__(self, *args, **kwargs) -> Any:
        """
        Performs evaluation for provided prediction and reference.
        """
        return self.evaluate(*args, **kwargs)


class MetadataAggregator(ABC):
    def __init__(self, remove_nulls: bool = False, debug: bool = False) -> None:
        self.debug = debug
        self.remove_nulls = remove_nulls

    @abstractmethod
    def aggregate(self, extractions: List[Dict[str, Any]]) -> Type[BaseModel]:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Type[BaseModel]:
        """
        Performs validation for the provided metadata.
        """
        return self.aggregate(*args, **kwargs)

    def _remove_nulls(self, item: T) -> T:
        """metadata extractions will often be missing values that the extractor couldn't locate
        in the chunk. this purges all the keys with null values to decrease token usage"""

        if isinstance(item, dict):
            filtered_dict = {
                k: self._remove_nulls(v)
                for k, v in item.items()
                if self._remove_nulls(v)
            }
            return filtered_dict if filtered_dict else None
        elif isinstance(item, list):
            filtered_list = [
                self._remove_nulls(v) for v in item if self._remove_nulls(v)
            ]
            return filtered_list if filtered_list else None
        elif isinstance(item, set):
            filtered_set = {
                self._remove_nulls(v) for v in item if self._remove_nulls(v)
            }
            return filtered_set if filtered_set else None
        elif isinstance(item, tuple):
            filtered_tuple = tuple(
                self._remove_nulls(v) for v in item if self._remove_nulls(v)
            )
            return filtered_tuple if filtered_tuple else None
        else:
            return item


def main():
    pass


if __name__ == "__main__":
    main()
