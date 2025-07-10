import json
from openai import OpenAI
import os
import re
import sys
sys.path.append("../")
from tqdm import tqdm


def landmark_extraction_prompting(env_id, split):
    client = OpenAI(api_key="OPENAI_API_KEYS")

    data_path = f"datasets/{env_id}/{split}.json"
    save_path = data_path.replace(f"{split}.json", f"{split}_lm.json")

    with open(data_path, 'r') as f:
        nav_tasks = json.load(f)['episodes']
    print(len(nav_tasks))
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            nav_tasks_with_landmarks = json.load(f)
    else:
        nav_tasks_with_landmarks = {'episodes': []}

    pre_count = len(nav_tasks_with_landmarks['episodes'])

    for i in tqdm(range(len(nav_tasks))):
        if i < pre_count:
            continue
        nav_task = nav_tasks[i]
        instruction = nav_task['instruction']['instruction_text']
        episode_id = nav_task['episode_id']

        try:
            completion = client.chat.completions.create(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a navigation aircraft, and now you need to navigate to a specified location according to a natural language instruction. You need to extract a landmark sequence from the instruction. The sequence order should be consistent with their appearance on the path. Your output should be in JSON format and must contain two fields: "Landmark sequence" and "Thought." "Landmark sequence" is your thinking result comprised of landmark phrases in the instruction. "Thought" is your thinking process.'
                    },
                    {
                        'role': 'user',
                        'content': f'The instruction is <{instruction}>'
                    }
                ],
                model='gpt-4o'
            )

            # print(instruction)
            message = completion.choices[0].message
            content = message.content
            # content = unicodedata.normalize('NFKC', message.content)

            # print(content)
            parsed_content = ""
            match = re.search(r'```json\s*(\{.*\})', content, re.DOTALL)
            if match:
                parsed_content = match.group(1)
                print(parsed_content)
            else:
                print("no match")
                pass
            res_json = json.loads(parsed_content)
            # print(res_json)
            nav_task['instruction']['landmarks'] = res_json['Landmark sequence']
            nav_task['instruction']['thought'] = res_json['Thought']

        except Exception as e:
            print(e)
            nav_task['instruction']['landmarks'] = []
            nav_task['instruction']['thought'] = ""
        nav_tasks_with_landmarks['episodes'].append(nav_task)
        with open(save_path, 'w') as f:
            json.dump(nav_tasks_with_landmarks, f, indent=4)

    with open(data_path.replace(f"{split}.json", f"{split}_lm.json"), 'w') as f:
        json.dump(nav_tasks_with_landmarks, f, indent=4)


def generate_image_description(image):

    data_uri = image_to_base64_data_uri(image)
    response = client.chat.completions.create(model='gpt-4o',
            messages=[{'role': 'system',
            'content': 'You are an assistant who perfectly describes images in urban environment, considering all your knowledge of urban environment.'
            }, {'role': 'user', 'content': [{'type': 'image_url',
            'image_url': {'url': data_uri}}, {'type': 'text',
            'text': 'Create a description about the main, distinct object in the bounding box with concise, up to eight-word sentence. Highlight its color, appearance, style, shape, structure, material.'
            }]}])


    print(response.choices[0].message.content)
    return response.choices[0].message.content


if __name__ == "__main__":
    val_seen_env_ids = [3, 17, 10, 12, 14, 5, 8, 2]
    val_unseen_env_ids = [7, 9, 13, 21, 24]

    for env_id in val_seen_env_ids:
        landmark_extraction_prompting(env_id=env_id, split="val_seen")

    for env_id in val_unseen_env_ids:
        landmark_extraction_prompting(env_id=env_id, split="val_unseen")
