#!/usr/bin/env python3

import re

from deepdiff import DeepSearch
from loguru import logger
from rapidfuzz import fuzz

from ..utils import flatten_dict
from ._base import MetadataEvaluator


class JaccardEvaluator(MetadataEvaluator):
    """
    This evaluator computes the ratio of tokens found between
    prediction and reference w.r.t overall tokens (prediction+reference)
    """

    def _evaluate(
        self,
        prediction: dict,
        reference: dict,
        **kwargs,
    ) -> float:
        prediction = " ".join(map(str, flatten_dict(prediction).values())).strip()
        prediction = prediction.lower() if self.ignore_case else prediction

        reference = " ".join(map(str, flatten_dict(reference).values())).strip()
        reference = reference.lower() if self.ignore_case else reference

        pred_tokens = set(prediction.split())
        ref_tokens = set(reference.split())
        common = pred_tokens.intersection(ref_tokens)

        if self.debug:
            logger.debug(f"Prediction Tokens :: {pred_tokens}")
            logger.debug(f"Reference Tokens :: {ref_tokens}")
            logger.debug(f"Common Tokens :: {common}")
        return 0 if not common else len(common) / len(pred_tokens.union(ref_tokens))


class FlattenedExactMatcher(MetadataEvaluator):
    """
    This evaluator first flattens everything to last node
    and for each node tries to compare the value from prediction
    to reference.
    """

    def _evaluate(
        self,
        prediction: dict,
        reference: dict,
        **kwargs,
    ) -> float:
        prediction = flatten_dict(prediction)
        reference = flatten_dict(reference)

        key_prep = lambda x: re.sub(r"\.\d+\.", ".", x)

        total_matches = 0
        _matched_keys = []
        for key in prediction:
            val = prediction[key]
            key = key_prep(key)
            matches = DeepSearch(reference, val, verbose_level=2).get(
                "matched_values",
                {},
            )
            if self.debug:
                logger.debug(f"Prediction key={key}, value={val}")
                logger.debug(matches)
                # pprint(matches, indent=2)
            for _m_key, _m_val in matches.items():
                _m_key_p = key_prep(_m_key)
                # if the prediction key matches with reference key,
                # count that and immediately halt.
                # No need to search for other value
                if key in _m_key_p:
                    _matched_keys.append(_m_key)
                    total_matches += 1
                    break
        if self.debug:
            logger.debug(f"Matched keys = {_matched_keys}")
        score = total_matches / len(prediction)
        score = min(1.0, score)
        return score


class RecursiveFuzzyMatcher(MetadataEvaluator):
    """
    This evalutor recursively traverses the dictionaries
    and for each key, at leaf node, does fuzzy matching.

    Algorithm how to:
        - Initially assign equal weights to each key present in prediction
        - For each key, recursively traverse the values
            - At leaf node, compute the fuzzy score
        - The final score for the key is the initial_weight * matched_score


    Args:
        ```remove_nulls```: ```bool```
            If enabled, all the empty/null values in the dict are removed
            recursively
        ```ignore_case```: ```bool```
            If enabled, matching is done ignoring the case
        ```threshold```: ```float```
            This is used to binarize the fuzzy score
        ```debug```: ```bool```
            If enabled, debug mode logs are printed.

    Usage

        .. code-blocka: python

            from larch.metadata.evaluators.RecursiveFuzzyMatcher

            gt_dict = ...
            pred_dict = ...

            res = RecursiveFuzzyMatcher(debug=True)(prediction=pred_dict, reference=gt_dict)
    """

    def __init__(
        self,
        remove_nulls: bool = True,
        ignore_case: bool = True,
        threshold: float = 0.75,
        debug: bool = False,
    ) -> None:
        super().__init__(
            remove_nulls=remove_nulls,
            ignore_case=ignore_case,
            debug=debug,
        )
        self.threshold = threshold

    @staticmethod
    def fuzzy_match_score(str1, str2, ignore_case: bool = True):
        if isinstance(str1, (bool, int, float)):
            str1 = str(str1)
        if isinstance(str2, (bool, int, float)):
            str2 = str(str2)
        if ignore_case:
            str1 = str1.lower()
            str2 = str2.lower()
        score = fuzz.ratio(str1, str2) / 100
        return score

    def compare_values(self, val1, val2, weight):
        # Check if both values are dictionaries
        if isinstance(val1, dict) and isinstance(val2, dict):
            return self.compare_dicts(val1, val2, weight)

        # Check if both values are lists
        elif isinstance(val1, list) and isinstance(val2, list):
            # Handle empty lists
            if not val1 or not val2:
                return 0

            # Compare each element in the lists
            scores = []
            for item1 in val1:
                item_scores = [self.compare_values(item1, item2, 1) for item2 in val2]
                # scores.append(max(item_scores) if item_scores else 0)
                scores.append(max(item_scores) > self.threshold if item_scores else 0)

            return sum(scores) / len(scores) * weight

        # Fuzzy matching at leaf level (both strings)
        else:
            return (
                RecursiveFuzzyMatcher.fuzzy_match_score(
                    val1,
                    val2,
                    ignore_case=self.ignore_case,
                )
                * weight
            )

    def compare_dicts(self, dict1, dict2, parent_weight=1):
        # Combine the keys from both dictionaries
        keys = set(dict1).union(dict2)
        if self.debug:
            logger.debug(f"Union keys | {keys}")

        total_score = 0
        weight_per_key = parent_weight / len(keys) if keys else 0

        # Recursively compare each key-value pair
        for key in keys:
            val1 = dict1.get(key, None)
            val2 = dict2.get(key, None)

            # Handle cases where either value is None
            if val1 is None or val2 is None:
                # Maybe assign a penalty for missing keys?
                continue
            total_score += self.compare_values(val1, val2, weight_per_key)

        return total_score

    def _evaluate(
        self,
        prediction: dict,
        reference: dict,
        **kwargs,
    ) -> float:
        return self.compare_dicts(prediction, reference)


def main():
    pass


if __name__ == "__main__":
    main()
