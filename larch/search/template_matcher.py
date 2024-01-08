from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI

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

    def match(self, query: str, top_k=1, **kwargs) -> List[str]:
        pass


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
        llm: Optional[BaseLanguageModel],
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
