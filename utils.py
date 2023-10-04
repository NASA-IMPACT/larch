#!/usr/bin/env python3


def is_lambda(obj) -> bool:
    return callable(obj) and obj.__name__ == "<lambda>"
