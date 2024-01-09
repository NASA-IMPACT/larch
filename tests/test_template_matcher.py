import pytest
from larch.search.template_matcher import FuzzySQLTemplateMatcher, SQLTemplate

class TestFuzzySQLTemplateMatcher:
    @pytest.fixture
    def templates(self):
        return [
            SQLTemplate(query_pattern="which what solutions agencies interested relate recommended",
                        sql_template="""SELECT DISTINCT {columns}
                        FROM metadata
                        JOIN solution ON metadata.id = solution.need_id
                        WHERE {comparison_key} ILIKE '%{comparing_value}%' OR similarity({comparison_key}, '{comparing_value}') > {similarity_threshold};"""
            ),
            SQLTemplate(query_pattern="How many solutions agencies interested relate recommended",
                        sql_template="""SELECT COUNT(DISTINCT {columns})
                        FROM metadata
                        JOIN solution ON metadata.id = solution.need_id
                        WHERE {comparison_key} ILIKE '%{comparing_value}%' OR similarity({comparison_key}, '{comparing_value}') >= {similarity_threshold};"""
            ),
        ]

    def test_match(self, templates):
        matcher = FuzzySQLTemplateMatcher(templates=templates, similarity_threshold=0.4)
        query1 = "Which agencies are interested in using the ICESat-2 satellite?"
        query2 = "How many solutions recommended the ICESat-2 satellite?"
        result1 = matcher.match(query1)
        result2 = matcher.match(query2)
        assert result1 == [templates[0].sql_template]
        assert result2 == [templates[1].sql_template]
