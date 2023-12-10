#!/usr/bin/env python3

import re
from abc import ABC, abstractmethod
from functools import cache
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from ..utils import remove_nulls

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

    Args:
        ```remove_nulls```: ```bool```
            If enabled, all the empty/null values in the dict are removed
            recursively
        ```ignore_case```: ```bool```
            If enabled, matching is done ignoring the case
        ```debug```: ```bool```
            If enabled, debug mode logs are printed.
    """

    def __init__(
        self,
        remove_nulls: bool = True,
        ignore_case: bool = True,
        debug: bool = False,
    ) -> None:
        self.remove_nulls = remove_nulls
        self.ignore_case = ignore_case
        self.debug = debug

    def evaluate(
        self,
        prediction: Union[Type[BaseModel], Dict],
        reference: Union[Type[BaseModel], Dict],
        **kwargs,
    ) -> Any:
        prediction = self._get_dict(prediction)
        prediction = remove_nulls(prediction) if self.remove_nulls else prediction
        reference = self._get_dict(reference)
        return self._evaluate(prediction=prediction, reference=reference, **kwargs)

    @abstractmethod
    def _evaluate(self, prediction: Dict, reference: Dict, **kwargs) -> Any:
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


def main():
    pass


if __name__ == "__main__":
    main()
