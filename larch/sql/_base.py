#!/usr/bin/env python3

import os
import sys
from abc import abstractmethod
from typing import Any, List, Optional

from .._base import AbstractClass
from ..schema import SQLTemplate


class SQLTemplateMatcher(AbstractClass):
    """
    SQLTemplateMatcher is a base class for all SQL based template matchers.
    """

    def __init__(
        self,
        templates: List[SQLTemplate],
        similarity_threshold: float = 0.4,
        debug: bool = False,
    ) -> None:
        self.templates = templates
        self.similarity_threshold = similarity_threshold
        self.debug = debug

    @abstractmethod
    def match(self, query: str, top_k=1, **kwargs) -> List[SQLTemplate]:
        """
        Match the given query against the templates.

        Args:
            query: The query to match against the templates.
            top_k: The number of top-k templates to return. Defaults to 1.
        Returns:
            A list of top-k templates that match the query with entity substitution.
        """
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> List[str]:
        return self.match(*args, **kwds)


def main():
    pass


if __name__ == "__main__":
    main()
