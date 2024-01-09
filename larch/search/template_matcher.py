from abc import ABC, abstractmethod

import re
from typing import Any, List, Optional
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from thefuzz import fuzz

from ..schema import SQLTemplate

class SQLTemplateMatcher(ABC):
    """
    SQLTemplateMatcher is a base class for all SQL based template matchers.
    """
    def __init__(self,
                templates: List[SQLTemplate],
                 similarity_threshold: float = 0.4,
                 debug: bool = False) -> None:
        self.templates = templates
        self.similarity_threshold = similarity_threshold
        self.debug=debug

    @abstractmethod
    def match(self, query: str, top_k=1, **kwargs) -> List[str]:
        """
        Match the given query against the templates.

        Args:
            query: The query to match against the templates.
            top_k: The number of top-k templates to return. Defaults to 1.
        Returns:
            A list of top-k templates that match the query with entity substitution.
        """
        raise NotImplementedError()


    def __call__(self, *args: Any, **kwds: Any) -> List[str]:
        return self.match(*args, **kwds)


class FuzzySQLTemplateMatcher(SQLTemplateMatcher):
    """
    FuzzySQLTemplateMatcher is a SQL based template matcher that uses fuzzy matching.
    Given a query, it will use rule-based matching to find best matching template
    and return the template(s) with entity substitution.

    Args:
        templates: A list of SQL templates.
        similarity_threshold: The similarity threshold to be used for fuzzy matching.
    """
    def __init__(self, templates: List[SQLTemplate],
                 similarity_threshold: float = 0.4,
                 debug: bool = False) -> None:
        super().__init__(templates=templates,
                         similarity_threshold=similarity_threshold,
                         debug = debug)
        
    def get_columns(self, query: str) -> List[str]:
        """
        Extracts columns from the user query.

        Args:
            query (str): User query string.

        Returns:
            List[str]: List of column names.
        """
        print("\n\n\n\n\n hellow world \n\n\n\n\n")
        columns = []
        # Use regular expression to identify keywords in the query
        match = re.search(
            r"(what|which|how many) (\w+)(?: and (\w+))?", query, re.IGNORECASE)
        if match:
            # Extract captured words from the regex match
            captured_words = match.groups()[1:]
            # Convert captured words to lowercase
            captured_words = [word.lower() for word in captured_words if word]
            print(f"Captured culumns: {captured_words}")
            for captured_word in captured_words:
                # Map captured words to corresponding SQL columns
                if captured_word == "agencies" or captured_word == "agency":
                    columns.append("metadata.organization_name")
                elif captured_word == "needs" or captured_word == "need":
                    columns.append("need.need_nature")
                elif captured_word == "satellite" or captured_word == "satellites":
                    columns.append("solution.platform")
                elif captured_word == "product" or captured_word == "products":
                    columns.append("solution.data_product")
                elif captured_word == "solution" or captured_word == "solutions":
                    columns.append("solution.id")
        print(f"returning columns: {columns}")
        return columns

    def get_comparison_key(self, comparison_value: str) -> str:
        """
        Determines the comparison key based on the comparison value.

        Args:
            comparison_value (str): Comparison value extracted from the user query.

        Returns:
            str: Comparison key.
        """
        # Map comparison values to corresponding SQL comparison keys
        if "satellite" in comparison_value or "mission" in comparison_value:
            return "solution.platform"
        elif "product" in comparison_value:
            return "solution.data_product"
        elif "observable" in comparison_value or "observe" in comparison_value:
            return "solution.observable"
        if "phenomenon" in comparison_value:
            return "need.phenomenon"

    def get_comparing_value(self, query: str) -> Optional[str]:
        """
        Extracts the comparing value from the user query.

        Args:
            query (str): User query string.

        Returns:
            Optional[str]: Comparing value or None if not found.
        """
        # Use regular expression to extract comparing value from the query
        match = re.search(
            r"(?:using|to use|use to|observe|are used|to observe|relate to|related to| to use| recommended| was| did)(.+?)\?", query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return None
        
    def fill_template(self, template: SQLTemplate, query: str) -> Optional[SQLTemplate]:
        """
        Fill the template with the given query.

        Args:
            template: The template to fill.
            query: The query to fill the template with.
        Returns:
            The filled template.
        """
        # Extract columns, comparing value, and comparison key from the user query
        columns = self.get_columns(query)
        comparing_value = self.get_comparing_value(query)
        comparison_key = self.get_comparison_key(comparing_value)
        # Check if both columns and comparing value are available
        if columns and comparing_value:
            # Substitute values into the SQL template
            args = {
                "columns": ", ".join(columns),
                "comparison_key": comparison_key,
                "comparing_value": comparing_value,
                "similarity_threshold": self.similarity_threshold
            }
            # Substitute the values into a copy of the template
            template = template.model_copy()
            template.sql_template = template.sql_template.format(**args)
            return template
        else:
            return None


    def match(self, query: str, top_k=1, **kwargs) -> List[str]:
        query = query.lower()
        # Calculate fuzzy match scores for each template
        match_scores = [fuzz.token_set_ratio(
            query, template.query_pattern.lower()) for template in self.templates]
        # Sort the index according to the best score
        match_scores = sorted(
            enumerate(match_scores), key=lambda x: x[1], reverse=True)
        # Get the best matches if the score is above the threshold
        best_matches = [self.templates[index] 
                        for index, score in match_scores
                          if score >= self.similarity_threshold]
        # Fill the template with the query
        best_matches = [self.fill_template(template, query)
                        for template in best_matches]
        # Remove None values from the list
        best_matches = [template for template in best_matches if template]
        
        return best_matches[:top_k]
        

class LLMBasedSQLTemplateMatcher(SQLTemplateMatcher):
    """
    LLMBasedSQLTemplateMatcher uses LLM to find the best matching template.
    Given a query, it will extract the key entities and use LLM to find best SQL template
    and generates a subsituted SQL query.

    Args:
        templates: A list of SQL templates.
        ddl_schema: The DDL schema for available tables.
        similarity_threshold: The similarity threshold to be used for fuzzy matching.
    """
    def __init__(self, templates: List[SQLTemplate],
                 llm: BaseLanguageModel,
                 ddl_schema: Optional[str] = None,
                 similarity_threshold: float = 0.4,
                 debug: bool = False) -> None:
        super().__init__(
            templates=templates,
            similarity_threshold=similarity_threshold,
            debug = debug)

        self.llm = llm or ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
        self.ddl_schema = ddl_schema

    def match(self, query: str, top_k = 1, **kwargs) -> List[str]:
        pass
