#!/usr/bin/env python3

import copy
import re
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple

from loguru import logger
from openai import BadRequestError
from pydantic import BaseModel, Field, create_model
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process

try:
    import spacy
except ImportError:
    logger.warning("spacy is not installed. Can't use `larch.processors.PIIRemover`")


from ._base import AbstractClass
from .metadata._base import AbstractMetadataExtractor


class TextProcessor(ABC):
    """
    A component for text processing.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @abstractmethod
    def process(self, text: str, **kwargs) -> str:
        return text

    def __call__(self, *args, **kwargs) -> str:
        return self.process(*args, **kwargs)


class PIIRemover(TextProcessor):
    """
    Rmoves Personal Identifiable Information
    """

    _PATTERN_EMAIL = re.compile(
        "(([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\."
        "([a-z]{2,6}(?:\.[a-z]{2})?))(?![^<]*>)",
    )
    _PATTERN_PHONE = re.compile(
        r"((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))",
    )

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.nlp = spacy.load("en_core_web_lg")

    def _process_name(self, text: str) -> str:
        doc = self.nlp(text)
        for entity in doc.ents:
            if (
                entity.label_ == "PERSON"
                and " " in entity.text
                and entity.text.istitle()
            ):  # Extracting Full Names by usig Spacy
                text = text.replace(entity.text, "<NAME>")
        return text

    def _process_email(self, text: str) -> str:
        return self._PATTERN_EMAIL.sub("<EMAIL_ID>", text)

    def _process_phone(self, text: str) -> str:
        return self._PATTERN_PHONE.sub("<PHONE_NUMBER>", text)

    def process(self, text: str) -> str:
        text = self._process_name(text)
        text = self._process_email(text)
        text = self._process_phone(text)
        return text


class NonAlphaNumericRemover(TextProcessor):
    """
    Removes non-alpha-numeric characters from the text.
    """

    _alnum_regex = re.compile(r"(?ui)\W")

    def __init__(self, ignore_case: bool = True, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.ignore_case = bool(ignore_case)

    def process(self, text: str) -> str:
        if self.ignore_case:
            text = text.lower()
        return self._alnum_regex.sub(" ", text).strip()


class TextProcessingPipeline(TextProcessor):
    """
    A container to hold provided text processors and execute serially.
    """

    def __init__(self, *processors, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.processors = processors

    def process(self, text: str) -> str:
        for processor in self.processors:
            text = processor(text)
        return text


class Matcher(AbstractClass):
    """
    A component to match text with a list of text.
    """

    def __init__(self, debug: bool = False, ignore_case: bool = True) -> None:
        super().__init__(debug=debug)
        self.ignore_case = ignore_case

    @abstractmethod
    def match(self, text: str, values: List[str]) -> List[Tuple[str, float]]:
        """
        Args:
            ```text```: ```str```
                Text that is to be matched
            ```values```: ```List[str]```
                A list of text to be matcheda against

        Returns:
            ```List[Tuple[str, float]]```
                A list of tuple that represent best matches.
                - First element of tuple is the matched text from the list of
                values
                - Second element of tuple is the matching score
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> List[tuple]:
        return self.match(*args, **kwargs)


class ExactMatcher(Matcher):
    """
    Use this if you want to match the texts exactly.
    (x==y)
    """

    def match(self, text: str, values: List[str]) -> List[Tuple[str, float]]:
        """
        Args:
            ```text```: ```str```
                Text that is to be matched
            ```values```: ```List[str]```
                A list of text to be matcheda against

        Returns:
            ```List[Tuple[str, float]]```
                A list of tuple that represent best matches.
                - First element of tuple is the matched text from the list of
                values
                - Second element of tuple is the matching score
        """
        text_org = text[:]
        if self.ignore_case:
            text = text.lower()
            values = list(map(str.lower, values))
        for val in values:
            if val == text:
                return [(text_org, 100.0)]
        return []


class FuzzyMatcher(Matcher):
    """
    This matches the text against the list of texts using fuzzy-matching.
    """

    def __init__(
        self,
        scorer=fuzz.WRatio,
        threshold: float = 0.95,
        ignore_case: bool = True,
    ) -> None:
        super().__init__(ignore_case=ignore_case)
        self.threshold = threshold
        self.scorer = fuzz.WRatio
        self.ignore_case = ignore_case

    def match(self, text: str, values: List[str]) -> List[tuple]:
        """
        Args:
            ```text```: ```str```
                Text that is to be matched
            ```values```: ```List[str]```
                A list of text to be matcheda against

        Returns:
            ```List[Tuple[str, float]]```
                A list of tuple that represent best matches.
                - First element of tuple is the matched text from the list of
                values
                - Second element of tuple is the matching score
        """
        if self.ignore_case:
            text = text.lower()
            values = list(map(str.lower, values))

        cutoff = self.threshold * 100
        matches = fuzz_process.extract(
            text,
            values,
            scorer=self.scorer,
            score_cutoff=cutoff,
        )
        return list(map(lambda x: (x[0], x[1]), matches))


class LLMMatcher(Matcher):
    """
    This is a LLM-based matcher that makes use of
    ```larch.metadata.AbstractMetadataExtractor```.

    The matching is treated as a metadata extraction problem via LLM.

    This creates a pydantic schema dynamically (at runtime) using
    provided `text` and `values`. Values represent the `whitelist` field
    of the schema and text is passed to the extractor to extract the matching
    string.

    Note:
        - Chances of not returning exactly the string present in the list
    """

    _PROMPT_WHITELIST = (
        "Value that is selected. If no value matches, just return empty string."
    )

    def __init__(
        self,
        extractor: AbstractMetadataExtractor,
        prompt_whitelist: Optional[str] = None,
        ignore_case: bool = False,
    ) -> None:
        """
        Args:
            ```extractor```: ```AbstractMetadataExtractor```
                Any metadata extractor within larch
            ```prompt_whitelist```: ```Optional[str]```
                Prompt that is use to descripe the `whitelist` field
                for the schema
        """
        super().__init__(ignore_case=ignore_case)
        self.extractor = extractor
        self.extractor.schema = None
        self.prompt_whitelist = prompt_whitelist or LLMMatcher._PROMPT_WHITELIST

    @property
    def schema(self) -> Optional[BaseModel]:
        return self.extractor.schema

    def match(self, text: str, values: List[str]) -> List[Tuple[str, float]]:
        """
        Args:
            ```text```: ```str```
                Text that is to be matched
            ```values```: ```List[str]```
                A list of text to be matcheda against

        Returns:
            ```List[Tuple[str, float]]```
                A list of tuple that represent best matches.
                - First element of tuple is the matched text from the list of
                values
                - Second element of tuple is the matching score
        """
        text_org = text[:]
        if self.ignore_case:
            text = text.lower()
            values = list(map(str.lower, values))

        # copy and modify schema at runtime
        extractor = copy.copy(self.extractor)

        extractor.schema = create_model(
            "_WhitelistSelector",
            whitelist=(
                Literal[tuple(values)],
                Field(
                    ...,
                    description=self.prompt_whitelist,
                ),
            ),
            score=(
                float,
                Field(
                    ...,
                    ge=0,
                    le=100,
                    description="Match score in range [0, 100]. If no match, return 0.0",
                ),
            ),
        )
        res = []
        try:
            match = extractor(text)
            if match.whitelist and match.whitelist != text:
                # Rant: Don't trust LLM scores out-of-box
                score = getattr(match, "score", 0.0)
                res = [(match.whitelist, score)]
        except (BadRequestError, AttributeError):
            res = []

        # set to initial None because schema is built at runtime here
        self.extractor.schema = None
        return res


class CombinedMatcher(Matcher):
    """
    This encapsulates all the matcher object in series,
    where we run each matching algorithm in waterfall,
    and the first one to match will be the final result.

    This is used when we want to combine all the matching algorithm,
    like a fallback approach.
    """

    def __init__(self, *matchers: Matcher) -> None:
        self.matchers = matchers

    def match(self, text: str, values: List[str]) -> List[Tuple[str, float]]:
        """
        Args:
            ```text```: ```str```
                Text that is to be matched
            ```values```: ```List[str]```
                A list of text to be matcheda against

        Returns:
            ```List[Tuple[str, float]]```
                A list of tuple that represent best matches.
                - First element of tuple is the matched text from the list of
                values
                - Second element of tuple is the matching score
        """
        for matcher in self.matchers:
            matches = matcher(text, values)
            if matches:
                return matches
        return []


def main():
    pass


if __name__ == "__main__":
    main()
