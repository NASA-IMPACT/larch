#!/usr/bin/env python3

import json
import multiprocessing
import re
from collections.abc import MutableMapping
from typing import Tuple, TypeVar

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, ValidationError
from langchain.schema import OutputParserException

T = TypeVar("T", bound=BaseModel)


def is_lambda(obj) -> bool:
    return hasattr(obj, "__name__") and callable(obj) and obj.__name__ == "<lambda>"


class PydanticOutputParserWithoutValidation(PydanticOutputParser):
    """
    A derivation of PydanticOutputParser without any validation.
    """

    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(
                r"\{.*\}",
                text.strip(),
                re.MULTILINE | re.IGNORECASE | re.DOTALL,
            )
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str, strict=False)
            return self.pydantic_object.model_construct(**json_object)

        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text)


def get_cpu_count() -> int:
    return multiprocessing.cpu_count()


def flatten_dict(dictionary: dict, parent_key=False, separator="."):
    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten_dict({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)
