import json
import os
import time
import base64
import unicodedata

from openai import OpenAI
from openai import AzureOpenAI

class LLM:
    def __init__(self, api_key, model_name, max_tokens, cache_name='default', **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.queried_tokens = 0

        cache_model_dir = os.path.join('llm', 'cache', self.model_name)
        os.makedirs(cache_model_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_model_dir, f'{cache_name}.json')
        self.cache = dict()

        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def query_api(self, prompt):
        raise NotImplementedError

    def get_cache(self, prompt, instance_idx):
        sequences = self.cache.get(instance_idx, [])

        for sequence in sequences:
            if sequence.startswith(prompt) and len(sequence) > len(prompt)+1:
                return sequence
        return None

    def add_to_cache(self, sequence, instance_idx):
        if instance_idx not in self.cache:
            self.cache[instance_idx] = []
        sequences = self.cache[instance_idx]

        # newest result to the front
        sequences.append(sequence)

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print('cache saved to: ' + self.cache_file)

    def get_sequence(self, prompt, instance_idx, read_cache=True):
        sequence = None
        if read_cache:
            sequence = self.get_cache(prompt, instance_idx)
        print('cached sequence')
        if sequence is None:
            print('query API')
            sequence = self.query_api(prompt)
            self.add_to_cache(sequence, instance_idx)
            #print('api sequence')
        return sequence


class OpenAI_LLM_v1(LLM):
    def __init__(self, model_name, api_key, client_type="openai", logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):

        if client_type == "openai":
            self.client = OpenAI(
                api_key=api_key,
            )
        elif client_type == "Azure":
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-07-01-preview",
                azure_endpoint="https://zhangweichen-3d-gpt4o.openai.azure.com/"
            )
        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_apis(self, prompt, image_paths=[], show_response=True):
        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if len(image_paths) > 0:
                for img_p in image_paths:
                    base64_image = self.encode_image(img_p)
                    query_messages["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })

            completion = self.client.chat.completions.create(
                messages=[
                    query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            # self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response

    def query_api(self, prompt, image_path=None,system=None, show_response=True):

        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_path is not None:
                base64_image = self.encode_image(image_path)
                query_messages["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            completion = self.client.chat.completions.create(
                messages=[
                    query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()

        except Exception as e:
            print(e)
            # self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response

    def query_api_map_gpt(self, prompt, system=None, image_path=None, show_response=False):

        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            sys_query_messages = {
                "role": "system",
                "content": system
            }

            completion = self.client.chat.completions.create(
                messages=[
                    sys_query_messages, query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            # self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        return response


class OpenAI_LLM_v2(LLM):
    def __init__(self, model_name, api_key, client_type="openai", logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):

        if client_type == "openai":
            self.client = OpenAI(
                api_key=api_key,
            )
        elif client_type == "Azure":
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-07-01-preview",
                azure_endpoint="Your end point"
            )
        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_viewpoint_api(self, prompt, image_paths=None, show_response=True):
        def query_func():
            content_block = []
            if image_paths is not None:
                for viewpoint, img_p in image_paths.items():
                    content_block.append({
                        "type": "text",
                        "text": f"{viewpoint} image: "
                    })
                    content_block.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.encode_image(img_p)}"
                        }
                    })
            content_block.append({
                "type": "text",
                "text": f"{prompt}"
            })

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content_block
                    }
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            # self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response

    def query_api(self, prompt, image_path=None,system=None, show_response=True):

        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_path is not None:
                base64_image = self.encode_image(image_path)
                query_messages["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            completion = self.client.chat.completions.create(
                messages=[
                    query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()

        except Exception as e:
            print(e)
            # self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response


