#!/usr/bin/env python3

import re
from pprint import pprint
from typing import Type, Union

from deepdiff import DeepSearch
from loguru import logger
from pydantic import BaseModel

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


class FlattenedExactMatch(MetadataEvaluator):
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


def main():
    pass


if __name__ == "__main__":
    main()
