import os
import re
import json
import openai
from dotenv import load_dotenv

class LLMTool:
    """
    A utility class for interacting with OpenAI's language models.

    This class handles initialization, sending prompts to the model, and processing the responses.
    """

    def __init__(self):
        """
        Initializes the LLMTool by loading environment variables and setting up the OpenAI client.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        org_id = os.getenv("OPENAI_ORG_ID")
        model = os.getenv("MODEL_NAME", "gpt-4o")
        self.client = openai.OpenAI(
            api_key=api_key,
            organization=org_id,
        )
        self.model = model
        print(f"LLMTool initialized with model: {self.model}")

    def run(self, system_prompt: str, user_prompt: str, screenshot_base64: str = None):
        """
        Sends a prompt to the OpenAI model and processes the response.

        Args:
            system_prompt (str): The system-level prompt to guide the model's behavior.
            user_prompt (str): The user-level prompt containing the main query or task.
            screenshot_base64 (str, optional): A base64-encoded screenshot to include in the prompt.

        Returns:
            str: The processed response from the model.
        """
        if screenshot_base64:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                    ]
                }
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        content = response.choices[0].message.content
        try:
            self.json_string = self._extract_json(content)
        except json.JSONDecodeError:
            print("Failed to decode JSON from response.")
            self.json_string = content
        except Exception as e:
            print(f"An error occurred: {e}")
            self.json_string = content

        print(f"LLMTool response: {self.json_string}")
        return self.json_string

    def _extract_json(self, content: str):
        """
        Extracts JSON data from a string, assuming it is enclosed in triple backticks.

        Args:
            content (str): The string containing JSON data.

        Returns:
            dict: The extracted JSON data as a Python dictionary.

        Raises:
            json.JSONDecodeError: If the JSON data cannot be decoded.
        """
        match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
        json_text = match.group(1) if match else content
        return json.loads(json_text)

    @staticmethod
    def postprocess_by_prompt(data):
        """
        Post-processes the JSON output from the LLM based on the task type.

        Args:
            data (dict or list): The JSON data returned by the LLM.

        Returns:
            dict or list: The processed data, formatted based on the task type.

        Supported task types:
        - PlanAction: Returns a list of steps with elements and app names.
        - EvaluateSubTaskAction: Returns a dictionary with status and advice.
        - Action classes: Returns a list of dictionaries for direct execution.
        """
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            if all(d.get("action_type") == "PlanAction" for d in data):
                return [{"step": i + 1, "element": d.get("element"), "app_name": d.get("app_name")} for i, d in enumerate(data)]
            if all("action_type" in d for d in data):
                return data  # executor output
        elif isinstance(data, dict):
            if data.get("action_type") == "EvaluateSubTaskAction":
                return {
                    "status": data.get("status", ""),
                    "advice": data.get("advice", "")
                }
        return data  # fallback
