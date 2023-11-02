#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Type, TypeVar

from loguru import logger
from pydantic import BaseModel

from ._base import MetadataValidator

T = TypeVar("T")


class SimpleInTextMetadataValidator(MetadataValidator):
    """
    This is a recursive validator that checks if the extracted text value
    in the metadata lies in the provided text.

    It recursively goes through each and every values and only performs
    the validation for the key that are to be validated

    Args:
        ```keys``` : ```Optional[List[str]]```
            List of keys to be matched in the metadata for which the validation
            is to be done
        ```debug``` : ```bool```
            Debug mode flag

    Usage

        .. code-blocka: python

            from larch.metadata import InstructorBasedOpenAIMetadataExtractor
            from larch.metadata.validators import SimpleInTextMetadataValidator

            metadata_extractor = InstructorBasedOpenAIMetadataExtractor(
                model="gpt-3.5-turbo-0613",
                schema=Metadata,
            )

            text = "Does MODIS have < 15 m resolution?"

            metadata = metadata_extractor(text)

            metadata = SimpleInTextMetadataValidator(
                keys=["spatio_temporal_resolutions"]
            )(res, text)
    """

    def __init__(self, keys: Optional[List[str]] = None, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.keys = set(keys or [])

    def validate(
        self,
        metadata: Type[BaseModel],
        text: str,
        **kwargs,
    ) -> Type[BaseModel]:
        """
        This runs the validation using the metadata and the text.

        Args:
            ```metadata```: ```BaseModel```
                Input metadata object
            ```text```: ```str```
                Input text in which the values are to be searched

        Returns:
            Final validated metadata object
        """
        dct = metadata.dict()
        try:
            dct_validated = self._recursive_validate(dct, text)
        except:
            logger.warning("Validation failed! Returning original metadata...")
            dct_validated = dct
        return metadata.model_construct(**dct_validated)

    def _recursive_validate(
        self,
        data: T,
        text,
        current_key: str = None,
    ) -> T:
        keys = self.keys
        if self.debug:
            logger.debug(f"Current key={current_key} | value = {data}")
        # check for dict
        if isinstance(data, dict):
            validated_dict = {}
            for key, value in data.items():
                validated_value = self._recursive_validate(value, text, key)
                if validated_value:
                    validated_dict[key] = validated_value
            return validated_dict
        # if list, recursively apply same thing to each item
        elif isinstance(data, list):
            res = map(
                lambda item: self._recursive_validate(item, text, current_key),
                data,
            )
            res = filter(None, res)
            return list(res)
        # final node is a text value
        elif isinstance(data, (str, int, float)) and current_key in keys:
            return (
                data
                if SimpleInTextMetadataValidator.is_in_text(str(data), text)
                else None
            )
        return data

    @staticmethod
    def is_in_text(text1: str, text2: str, lowercase: bool = True) -> bool:
        text1, text2 = str(text1), str(text2)
        text1 = text1.lower() if lowercase else text1
        text2 = text2.lower() if lowercase else text2
        return text1 in text2
