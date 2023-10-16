#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel

from ._base import MetadataValidator


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
            dct_validated = self.recursive_validate_dict(dct, text)
        except:
            logger.warning("Validation failed! Returning original metadata...")
            dct_validated = dct
        return metadata.model_construct(**dct_validated)

    def recursive_validate_dict(
        self,
        dct: Dict[str, Any],
        text,
        current_key: str = None,
    ) -> dict:
        keys = self.keys
        if self.debug:
            logger.debug(f"Current key={current_key} | value = {dct}")
        if isinstance(dct, dict):
            validated_dct = {}
            for key, value in dct.items():
                current_key = key
                validated_value = value
                if isinstance(value, (dict, list)):
                    validated_value = self.recursive_validate_dict(
                        value,
                        text,
                        current_key,
                    )
                if isinstance(value, str) and current_key in keys:
                    validated_value = (
                        value
                        if SimpleInTextMetadataValidator.is_in_text(value, text)
                        else None
                    )
                validated_dct[
                    key
                ] = validated_value  # if current_key in keys else value
            return validated_dct

        elif isinstance(dct, list) and current_key in keys:
            validated_list = [
                item
                for item in dct
                if SimpleInTextMetadataValidator.is_in_text(item, text)
            ]
            return validated_list
        elif isinstance(dct, list) and current_key not in keys:
            return dct
        return dct if SimpleInTextMetadataValidator.is_in_text(dct, text) else None

    @staticmethod
    def is_in_text(text1: str, text2: str, lowercase: bool = True) -> bool:
        text1, text2 = str(text1), str(text2)
        text1 = text1.lower() if lowercase else text1
        text2 = text2.lower() if lowercase else text2
        return text1 in text2
