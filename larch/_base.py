#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Optional


class AbstractClass(ABC):
    def __init__(self, name: Optional[str] = None, debug: bool = False) -> None:
        self.name = name
        self.debug = bool(debug)

    def __classname__(self) -> str:
        return self.name or self.__class__.__name__
