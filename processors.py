#!/usr/bin/env python3

import re
from abc import ABC, abstractmethod

import spacy


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


def main():
    pass


if __name__ == "__main__":
    main()
