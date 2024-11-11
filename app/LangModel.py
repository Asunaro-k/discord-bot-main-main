import os
import requests
import json
from collections.abc import Generator

class LangModel:
    def __init__(self, api_key:str, api_url:str, model_name:str):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _generate(self, prompt:str, stream:bool) -> str:
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
        }
        response = requests.post(
            f"{self.api_url}/generate", headers=self.headers, json=data)
        return response
        
    def generate(self, prompt:str) -> str:
        """complate the prompt and return the response"""
        response = self._generate(prompt, False)
        if response.status_code == 200:
            response_data = response.json()
            if response_data['response'] == "" or response_data['response'] == None:
                ValueError(f"Error: {response_data['error']}")
            return response_data['response']
        else:
            ValueError(f"Error: {response.status_code}, {response.text}")

    def stream_generate(self, prompt:str) -> Generator[dict[str, str], None, None]:
        """complate the prompt and return the response"""
        response = self._generate(prompt, True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    response_data = line.decode('utf-8')
                    response_data:dict[str, str] = json.loads(response_data)
                    yield response_data
        else:
            ValueError(f"Error: {response.status_code}, {response.text}")

    def _chat(self, messages:list[dict[str, str]], stream:bool) -> str:
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        response = requests.post(f"{self.api_url}/chat", headers=self.headers, json=data)
        return response

    def chat(self, messages:list[dict[str, str]]) -> str: 
        response = self._chat(messages, False)
        if response.status_code == 200:
                response_data = response.json()
                return response_data['message']
        else:
            ValueError(f"Error: {response.status_code}, {response.text}")

    def stream_chat(
            self, 
            messages:list[dict[str, str]]) -> Generator[dict[str, str], None, None]:
        response = self._chat(messages, True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    response_data = line.decode('utf-8')
                    response_data:dict[str, str] = json.loads(response_data)
                    yield response_data
        else:
            ValueError(f"Error: {response.status_code}, {response.text}")

if __name__ == '__main__':
    def value_from_env():
        api_key = os.environ['OLLAMA_API_KEY']
        api_url = os.environ['OLLAMA_URL']

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return api_key, api_url, headers
    api_key, api_url, headers = value_from_env()
    lm = LangModel(api_key, api_url, "gemma2:9b")

    def test_default():
        print(lm.generate("なぜ空は青いのですか?"))
        print(lm.chat([
            {
                "role": "system",
                "content": "あなたは知識豊富なアシスタントです。"
            },
            {
                "role": "user",
                "content": "なぜ空は青いのですか?"
            },
            {
                "role": "assistant",
                "content": "空が青い理由は、太陽光が大気中の小さな分子によって散乱されるからです。青い光は波長が短いため、他の色よりも強く散乱されます。"
            },
            {
                "role": "user",
                "content": "夕焼けが赤いのはなぜですか?"
            }
        ]))
    # test_default()

    def test_stream():
        for response in lm.generate("なぜ空は青いのですか?", stream=True):
            print(response)
        messages = [
            {
                "role": "system",
                "content": "あなたは知識豊富なアシスタントです。"
            },
            {
                "role": "user",
                "content": "なぜ空は青いのですか?"
            },
            {
                "role": "assistant",
                "content": "空が青い理由は、太陽光が大気中の小さな分子によって散乱されるからです。青い光は波長が短いため、他の色よりも強く散乱されます。"
            },
            {
                "role": "user",
                "content": "夕焼けが赤いのはなぜですか?"
            }
        ]
        # for response in lm.chat(messages, stream=True):
        #     print(response)
    # test_stream()