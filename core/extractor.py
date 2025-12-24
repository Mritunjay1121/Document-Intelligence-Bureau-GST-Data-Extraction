from openai import OpenAI
import json
from typing import Dict, Any, List


class ExtractionService:
    """extract parameters using GPT-4"""

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4-turbo-preview"

    def extract_parameter(self, parameter_name, parameter_description, context, source_location):
        """
        extract single parameter using GPT-4
        """
        # build the prompt
        prompt = f"""Extract the following parameter from the document:

PARAMETER: {parameter_name}
DESCRIPTION: {parameter_description}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
1. Extract the exact value (no formatting - remove commas, currency symbols)
2. If not found, return null
3. Be precise - no approximations

Return ONLY a JSON object:
{{
    "value": <number or string or null>,
    "source": "{source_location}"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting financial data from documents. Extract exact values without any formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            return {
                "value": None,
                "source": f"Error: {str(e)}"
            }

    def extract_from_table(self, parameter_name, tables):
        """
        extract parameter from tables
        TODO: maybe optimize this for large tables
        """
        # convert tables to text format
        tables_text = ""
        table_count = 0
        for i, table in enumerate(tables):
            if table_count >= 5:  # limit to 5 tables
                break

            tables_text += f"\n[Table {i+1} - Page {table['page']}]\n"

            headers = table.get("headers", [])
            rows = table.get("rows", [])

            # add headers
            if headers:
                header_str = ""
                for h in headers:
                    header_str += str(h) + " | "
                tables_text += header_str + "\n"

            # add rows (max 20 per table)
            row_count = 0
            for row in rows:
                if row_count >= 20:
                    break

                row_str = ""
                for c in row:
                    if c:
                        row_str += str(c) + " | "
                    else:
                        row_str += " | "
                tables_text += row_str + "\n"
                row_count += 1

            table_count += 1

        # build prompt
        prompt = f"""Find '{parameter_name}' in these tables:

{tables_text}

Return JSON:
{{
    "value": <extracted value without formatting>,
    "source": "<table number and location>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract data from tables. Return exact numbers without formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            return {
                "value": None,
                "source": f"Error: {str(e)}"
            }


