#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SQLTemplate:
    """
    SQLTemplate represents a SQL template for a given query pattern.

    E.g.:
    query_pattern: "When did <mission_name> launch?"
    sql_template: "SELECT date_operational FROM  <table_name> WHERE mission_name ILIKE '%<mission_name>%';"
    """

    @dataclass
    class Example:
        """
        Example represents an example related to the SQL template.

        Attributes:
        - query (str): The query example.
        - sql (str): The corresponding SQL example.
        - result (Any): The expected result of the example.
        """

        query: str
        sql: str
        result: Any

    query_pattern: str
    sql_pattern: str
    examples: Optional[List[Example]] = None
    intent: Optional[str] = None
    description: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None
