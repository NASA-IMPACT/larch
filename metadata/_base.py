#!/usr/bin/env python3

import re
from abc import ABC, abstractmethod
from functools import cache
from typing import Callable, Optional, Type

from pydantic import BaseModel


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
        text = self._preprocess_text(text)
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


def main():
    pass


if __name__ == "__main__":
    main()
