#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process
from rapidfuzz import utils as fuzz_utils

from ..processors import NonAlphaNumericRemover, TextProcessor
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

    def _validate(
        self,
        metadata: dict,
        text: str,
        **kwargs,
    ) -> dict:
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
        return self._recursive_validate(metadata, text)

    def _recursive_validate(
        self,
        data: Any,
        text,
        current_key: str = None,
    ) -> Any:
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


class WhitelistBasedMetadataValidator(MetadataValidator):
    """
    This validator uses a whitelist to standardize values in a field
    in the metadata.
    Each field has a certain value and each value could take on different
    alternate values during extraction process.
    This validator recursively traverses the whitelist to figure out
    (using fuzzy matching) which standard value to inject.

    Args:
        ```whitelists```: ```Dict[str, Dict[str, List[str]]]```
            A dictionary mapping for whitelist.
        ```fuzzy_threshold```: ```float```
            Fuzzy match threshold. matches >= fuzzy_threshold will be
            used for standardization
        ```debug```: ```bool```
            Debug mode flag

    Note:
        whitelists is of structure:

            .. code-block: python

                whitelists = {
                    <field_1>: {
                    <standard_value_1_str>: [<alternate_value_1>, <alternate_value_2>]
                    <standard_value_2_str>: [<alternate_value_1>, <alternate_value_2>]
                    },
                    <field_2>: {
                    ...
                    }
                }

                whitelists = {'observable': {'aerosols': ['aerosols',
                   'aerosol optical depth',
                   'aerosol extinction',
                   'AOD',
                   'aerosol concentration',
                   'pm2.5',
                   'particulate matter',
                   'Aerosol vertical distribution']
                   }
               }

    Usage:

        .. code-block: python

            from larch.metadata.validators import WhitelistBasedMetadataValidator
            from larch.processors import NonAlphaNumericRemover

            metadata = <pydantic_object>
            # or
            metadata = <dict>
            validator = WhitelistBasedMetadataValidator(
                whitelists=whitelists,
                fuzzy_threshold=0.90,
                fuzzy_scorer=fuzz.WRatio,
                text_processor=NonAlphaNumericRemover(),
                debug=False,
            )
            metadata_validated = validator(metadata)
    """

    def __init__(
        self,
        whitelists: Dict[str, Dict[str, List[str]]],
        fuzzy_threshold: float = 0.95,
        fuzzy_scorer: Callable = fuzz.WRatio,
        text_processor: Optional[Union[Callable, TextProcessor]] = None,
        ignore_case: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug, ignore_case=ignore_case)
        self.whitelists = whitelists
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_scorer = fuzzy_scorer or fuzz.WRatio

        # default: remove non-alpha-numeric values
        self.text_processor = text_processor or NonAlphaNumericRemover(
            ignore_case=ignore_case,
        )

    def _validate(self, metadata: dict, **kwargs) -> dict:
        return self._recursive_validate(metadata, current_key=None)

    def _recursive_validate(
        self,
        data: Any,
        current_key: str = None,
    ) -> Any:
        """
        This recursively traverses the dictionary to do the standardization
        """
        # check for dict
        if isinstance(data, dict):
            validated_dict = {}
            for key, value in data.items():
                validated_value = self._recursive_validate(value, key)
                if validated_value:
                    validated_dict[key] = validated_value
            return validated_dict

        # if list, recursively apply same thing to each item
        elif isinstance(data, list):
            res = map(
                lambda item: self._recursive_validate(item, current_key),
                data,
            )
            res = filter(None, res)
            return list(res)

        # final leaf/node is a text value
        elif isinstance(data, (str, int, float)) and current_key is not None:
            return self.standardize_value(
                self.whitelists,
                current_key,
                data,
                fuzzy_threshold=self.fuzzy_threshold,
                scorer=self.fuzzy_scorer,
                processor=self.text_processor,
                debug=self.debug,
            )

        return data

    @staticmethod
    def _has_word(text: str, values: List[str]) -> bool:
        # avoid matching shorter text
        if len(text) < 3:
            return False

        text = text.lower()
        for val in values:
            if (text in val) or (val in text):
                return True
        return False

    @staticmethod
    def _exact_match(text: str, values: List[str]) -> bool:
        """
        If the value exactly matches, return a match
        """
        # avoid matching shorter text
        if len(text) < 3:
            return False

        text = text.lower()
        for val in values:
            if val == text:
                return True
        return False

    @staticmethod
    def standardize_value(
        whitelists: dict,
        field_name: str,
        extracted_value: str,
        fuzzy_threshold: float = 0.95,
        scorer: Callable = fuzz.WRatio,
        processor: Callable = None,
        debug: bool = False,
    ) -> str:
        field_dct = whitelists.get(field_name, {}).copy()
        if not field_dct:
            return extracted_value

        cutoff = fuzzy_threshold * 100
        extracted_value_processed = (
            processor(extracted_value) if processor is not None else extracted_value
        )

        best_matches = []
        for key, values in field_dct.items():
            values = list(map(processor, values))
            fuzz_matches = fuzz_process.extract(
                extracted_value_processed,
                values,
                scorer=scorer,
                score_cutoff=cutoff,
            )
            _word_match = WhitelistBasedMetadataValidator._exact_match(
                extracted_value_processed,
                values,
            )
            if _word_match:
                best_matches.append((key, extracted_value, 100.0))
            elif fuzz_matches:
                best_matches.extend([(key, fm[0], fm[1]) for fm in fuzz_matches])
        if debug and best_matches:
            logger.debug(
                f"extracted value={extracted_value} | field={field_name} | best_matches={best_matches}",
            )

        best_matches = sorted(best_matches, key=lambda x: x[-1], reverse=True)
        return best_matches[0][0] if best_matches else extracted_value
