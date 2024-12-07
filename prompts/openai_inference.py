import json
from openai import OpenAI

"""
`api_key.json` should be a JSON file with the following structure:
{
    "base_url": "https://api.openai.com",   # Or any other base URL
    "api_key": "sk-..."
}
"""

class Assistant:
    def __init__(self):
        with open("api_key.json", "r") as f:
            api_key_info = json.load(f)

        base_url = api_key_info["base_url"]
        api_key = api_key_info["api_key"]

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def inference(self, prompt, temperature=0.5, max_tokens=4096):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

