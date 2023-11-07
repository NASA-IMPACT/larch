#!/usr/bin/env python3

import json
import multiprocessing
import re
from collections.abc import MutableMapping
from typing import List, TypeVar

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, ValidationError
from langchain.schema import OutputParserException

from .structures import Document

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


def remove_nulls(data: T, null_vals=[None, "N/A", "unknown"]) -> T:
    """
    Remove empty nodes from the input dictionary
    """
    null_strs = list(map(str, null_vals))
    null_strs += list(map(str.lower, null_strs))
    null_strs += list(map(str.upper, null_strs))
    null_strs += list(map(str.capitalize, null_strs))
    null_vals = list(set(null_vals + null_strs))

    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            cleaned_value = remove_nulls(value, null_vals)
            if cleaned_value not in null_vals and cleaned_value:
                new_dict[key] = cleaned_value
        return new_dict

    elif isinstance(data, list):
        new_list = []
        for item in data:
            cleaned_item = remove_nulls(item, null_vals)
            if cleaned_item not in null_vals and cleaned_item:
                new_list.append(cleaned_item)

        return new_list

    else:
        return data


def remove_duplicate_documents(documents: List[Document]) -> List[Document]:
    """
    remove_duplicate_documents checks for duplicate texts in documents
    and returns a list of unique documents.
    """
    _checks = set()
    unique_documents = []
    for document in documents:
        text = re.sub(r"\s+", "", document.text)
        if text not in _checks:
            unique_documents.append(document)
            _checks.add(text)
    return unique_documents
