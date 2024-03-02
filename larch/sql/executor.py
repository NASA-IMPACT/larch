#!/usr/bin/env python3
import re
from typing import Any, Dict, List

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.sql import text as SQLText

from .._base import AbstractClass


class SQLExecutor(AbstractClass):
    """
    Minimal class to execute direct sql statements.
    """

    def __init__(self, db_uri: str, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.engine = create_engine(db_uri)

    def execute_raw_sql(self, statement: str) -> List[Dict[str, Any]]:
        statement = statement.strip(";")
        res = []
        if self.is_unsafe(statement):
            raise ValueError("Unsafe SQL statement. Aborting execution!")
        if self.debug:
            logger.debug(f"Executing SQL Statement :: {statement}")
        with self.engine.connect() as conn:
            res = conn.execute(SQLText(statement))
            columns = res.keys()
            results = [
                {column: value for column, value in zip(columns, row)}
                for row in res.fetchall()
            ]
        return results

    def is_unsafe(self, statement: str) -> bool:
        patterns = [
            r"(;|--)",  # multiple statements or comments
            r"(\bDROP\b|\bALTER\b)",  # DDL operations
            r"\bEXEC(\s|\()|\bEXECUTE\b",  # executing stored procedures
        ]
        for pattern in patterns:
            if re.search(pattern, statement, re.IGNORECASE):
                return True
        return False

    def __call__(self, *args, **kwargs):
        return self.execute_raw_sql(*args, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
