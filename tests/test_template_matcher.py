# ignore warnings
import warnings

import pytest

from larch.search.template_matcher import FuzzySQLTemplateMatcher, SQLTemplate


class TestFuzzySQLTemplateMatcher:
    """Test the FuzzySQLTemplateMatcher class."""

    @pytest.fixture
    def templates(self):
        return [
            SQLTemplate(
                query_pattern=r"Which agencies are interested in using the (?P<key1>.+) satellite?",
                sql_pattern="SELECT DISTINCT metadata.organization_name FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%{key1}%' OR similarity(solution.platform, '{key1}') > {similarity_threshold};",
                examples=[
                    SQLTemplate.Example(
                        query="Which agencies are interested in using the ICESat-2 satellite?",
                        sql="SELECT DISTINCT metadata.organization_name FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%ICESat-2%' OR similarity(solution.platform, 'ICESat-2') > 0.4;",
                        result=[
                            "Department of the Interior",
                            "Environmental Protection Agency (EPA)",
                        ],
                    ),
                ],
                intent="Retrieve information about which agencies are interested in using a specific satellite.",
            ),
            SQLTemplate(
                query_pattern=r"How many agencies were recommending to use the (?P<key1>.+) satellite?",
                sql_pattern="SELECT COUNT(DISTINCT metadata.organization_name) FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%{key1}%' OR similarity(solution.platform, '{key1}') >= {similarity_threshold};",
                examples=[
                    SQLTemplate.Example(
                        query="How many agencies were recommending to use the ICESat-2 satellite?",
                        sql="SELECT COUNT(DISTINCT metadata.organization_name) FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%ICESat-2%' OR similarity(solution.platform, 'ICESat-2') >= 0.4;",
                        result=2,
                    ),
                ],
                intent="Retrieve count of agencies recommending the use of a specific satellite.",
            ),
            SQLTemplate(
                query_pattern=r"Which organizations are interested in using the (?P<key1>.+) or (?P<key2>.+) satellites?",
                sql_pattern="SELECT DISTINCT metadata.organization_name FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%{key1}%' OR solution.platform ILIKE '%{key2}%';",
                examples=[
                    SQLTemplate.Example(
                        query="Which organizations are interested in using the ICESat-2 or LandSat-8 satellites?",
                        sql="SELECT DISTINCT metadata.organization_name FROM metadata JOIN solution ON metadata.id = solution.need_id WHERE solution.platform ILIKE '%ICESat-2%' OR solution.platform ILIKE '%LandSat-8%';",
                        result=[
                            "Department of the Interior",
                            "Environmental Protection Agency (EPA)",
                        ],
                    ),
                ],
                intent="Retrieve information about which agencies are interested in using a specific satellites.",
            ),
        ]

    def test_match_with_single_key(self, templates):
        """Test the match method of FuzzySQLTemplateMatcher with a single key."""
        matcher = FuzzySQLTemplateMatcher(templates=templates, similarity_threshold=0.4)
        # Test the first template
        query1 = templates[0].examples[0].query
        sql1 = templates[0].examples[0].sql
        result1 = matcher.match(query1)
        assert result1[0].sql_pattern == sql1

    def test_match_with_multiple_keys(self, templates):
        """Test the match method of FuzzySQLTemplateMatcher with multiple keys."""
        matcher = FuzzySQLTemplateMatcher(templates=templates, similarity_threshold=0.4)
        # Test the third template
        query3 = templates[2].examples[0].query
        sql3 = templates[2].examples[0].sql
        result3 = matcher.match(query3)
        assert result3[0].sql_pattern == sql3
