import unittest

from core.utils import parse_sql_from_string


class SQLExtractionTests(unittest.TestCase):
    def test_last_fenced_query_remains_preferred(self):
        response = "```sql\nSELECT 1;\n```\nThen:\n```sql\nSELECT 2;\n```"
        self.assertEqual(parse_sql_from_string(response), "SELECT 2;")

    def test_plain_multiline_query_preserves_literals(self):
        sql = "SELECT `Abbreviation`\nFROM `airlines`\nWHERE `Airline` = 'JetBlue Airways';"
        self.assertEqual(parse_sql_from_string(" \n" + sql + "\n"), sql)

    def test_plain_cte_and_lowercase_query(self):
        for sql in ("WITH x AS (SELECT 1 AS n) SELECT n FROM x;", "select 1;"):
            with self.subTest(sql=sql):
                self.assertEqual(parse_sql_from_string(sql), sql)

    def test_rejects_prose_non_queries_and_broken_fences(self):
        for response in ("Here is SELECT 1;", "DELETE FROM singer;", "```sql\nSELECT 1;", ""):
            with self.subTest(response=response):
                self.assertEqual(
                    parse_sql_from_string(response),
                    "error: No SQL found in the input string",
                )


if __name__ == "__main__":
    unittest.main()
