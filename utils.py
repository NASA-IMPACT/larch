#!/usr/bin/env python3

import json
import re
from typing import TypeVar

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, ValidationError
from langchain.schema import OutputParserException

T = TypeVar("T", bound=BaseModel)


def is_lambda(obj) -> bool:
    return callable(obj) and obj.__name__ == "<lambda>"


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
