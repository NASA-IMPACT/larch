import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from rapidfuzz import fuzz

from ..schema import SQLTemplate


class SQLTemplateMatcher(ABC):
    """
    SQLTemplateMatcher is a base class for all SQL based template matchers.
    """

    def __init__(
        self,
        templates: List[SQLTemplate],
        similarity_threshold: float = 0.4,
        debug: bool = False,
    ) -> None:
        self.templates = templates
        self.similarity_threshold = similarity_threshold
        self.debug = debug

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

    def __init__(
        self,
        templates: List[SQLTemplate],
        similarity_threshold: float = 0.4,
        debug: bool = False,
    ) -> None:
        super().__init__(
            templates=templates,
            similarity_threshold=similarity_threshold,
            debug=debug,
        )

    def fill_template(self, template: SQLTemplate, query: str) -> Optional[SQLTemplate]:
        """
        Fill the template with the query.

        Args:
            template: The SQL template to fill.
            query: The query to fill the template with.
        Returns:
            The filled template if the query matches the template, otherwise None.
        """
        # Extract the entities from the query
        pattern = re.compile(template.query_pattern, re.IGNORECASE)
        match = pattern.match(query)
        if match:
            # Make a copy of the template to avoid modifying the original template
            template = template.model_copy()
            comparing_value = match.groups()[0]
            # Substitute the entities in the SQL template
            template.sql_pattern = template.sql_pattern.format(
                comparing_value=comparing_value,
                similarity_threshold=self.similarity_threshold,
            )
            return template
        else:
            return None

    def match(self, query: str, top_k=1, **kwargs) -> List[str]:
        query = query.lower()
        # Calculate fuzzy match scores for each template
        match_scores = [
            fuzz.token_set_ratio(
                query,
                template.query_pattern.lower(),
            )
            for template in self.templates
        ]
        # Sort the index according to the best score
        match_scores = sorted(
            enumerate(match_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        # Get the best matches if the score is above the threshold
        best_matches = [
            self.templates[index]
            for index, score in match_scores
            if score >= self.similarity_threshold
        ]
        # Fill the template with the query
        best_matches = [
            self.fill_template(template, query) for template in best_matches
        ]
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

    def __init__(
        self,
        templates: List[SQLTemplate],
        llm: BaseLanguageModel,
        ddl_schema: Optional[str] = None,
        similarity_threshold: float = 0.4,
        debug: bool = False,
    ) -> None:
        super().__init__(
            templates=templates,
            similarity_threshold=similarity_threshold,
            debug=debug,
        )

        self.llm = llm or ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
        self.ddl_schema = ddl_schema

    def match(self, query: str, top_k=1, **kwargs) -> List[str]:
        pass
