# ignore warnings
import warnings

import pytest

from larch.search.template_matcher import FuzzySQLTemplateMatcher, SQLTemplate

warnings.filterwarnings("ignore")


class TestFuzzySQLTemplateMatcher:
    """Test the FuzzySQLTemplateMatcher class."""

    @pytest.fixture
    def templates(self):
        return [
            SQLTemplate(
                query_pattern=r"Which agencies are interested in using the (\S+) satellite?",
                sql_pattern="SELECT DISTINCT metadata.organization_name FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%{comparing_value}%' OR similarity(solution.platform, '{comparing_value}') > {similarity_threshold};",
                examples=[
                    SQLTemplate.Example(
                        query="Which agencies are interested in using the ICESat-2 satellite?",
                        sql="SELECT DISTINCT metadata.organization_name FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%icesat-2%' OR similarity(solution.platform, 'icesat-2') > 0.4;",
                        result=[
                            "Department of the Interior",
                            "Environmental Protection Agency (EPA)",
                        ],
                    ),
                ],
                intent="Retrieve information about which agencies are interested in using a specific satellite.",
            ),
            SQLTemplate(
                query_pattern=r"How many agencies were recommending to use the (\S+) satellite?",
                sql_pattern="SELECT COUNT(DISTINCT metadata.organization_name) FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%{comparing_value}%' OR similarity(solution.platform, '{comparing_value}') >= {similarity_threshold};",
                examples=[
                    SQLTemplate.Example(
                        query="How many agencies were recommending to use the ICESat-2 satellite?",
                        sql="SELECT COUNT(DISTINCT metadata.organization_name) FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%icesat-2%' OR similarity(solution.platform, 'icesat-2') >= 0.4;",
                        result=2,
                    ),
                ],
                intent="Retrieve count of agencies recommending the use of a specific satellite.",
            ),
        ]

    def test_match(self, templates):
        """Test the match method of FuzzySQLTemplateMatcher."""
        matcher = FuzzySQLTemplateMatcher(templates=templates, similarity_threshold=0.4)
        # Test the first template
        query1 = templates[0].examples[0].query
        sql1 = templates[0].examples[0].sql
        result1 = matcher.match(query1)
        assert result1[0].sql_pattern == sql1
        # Test the second template
        query2 = templates[1].examples[0].query
        sql2 = templates[1].examples[0].sql
        result2 = matcher.match(query2)
        assert result2[0].sql_pattern == sql2
