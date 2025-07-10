import os
import re
import sys
import time
import json
from pathlib import Path
sys.path.append(str(Path(str(os.getcwd())).resolve()))

from src.llm.query_llm import OpenAI_LLM, Qwen_VL
from src.vlnce_src.env import AirVLNENV

from secret.keys import OPENAI_API_KEYS, DASHSCOPE_API_KEY
from airsim_plugin.airsim_settings import DefaultAirsimActionCodes, ObservationDirections


class TaskParser:
    def __init__(self, llm):
        self.llm = llm

    def navigation_parsing(self, navigation_task):
        parsed_subtask_sequences = []

        sub_tasks = self.coarse_task_parsing(navigation_task)

        # subtask corpus to commands and objects
        parsed_subtask_corpuses = []
        for sub_task in sub_tasks:
            parsed_subtask_corpus = self.sub_task_parsing(sub_task)
            parsed_subtask_corpuses.append(parsed_subtask_corpus)

        # commands to actions
        for sub_corpus in parsed_subtask_corpuses:
            atom_commands = []
            atom_commands_idxes = []

            atom_objects = []
            atom_objects_idxes = []

            atom_corpuses = sub_corpus.split(";")
            for i, atom_corpus in enumerate(atom_corpuses):
                atom, label = atom_corpus.split(",")

                if label == '0':
                    atom_commands.append(atom)
                    atom_commands_idxes.append(i)
                elif label == '1':
                    atom_objects.append(atom)
                    atom_objects_idxes.append(i)
                else:
                    print("unknown label type: {}".format(label))

            actions = self.action_mapping(atom_commands)
            if not len(actions) == len(atom_commands):
                print("Warning: translated actions are not aligned with input commands")
                return None

            parsed_subtask_sequence = [[] for _ in range(len(atom_corpuses))]
            pt1 = 0
            pt2 = 0
            for idx in range(len(parsed_subtask_sequence)):
                if pt1 < len(atom_commands_idxes) and idx == atom_commands_idxes[pt1]:
                    parsed_subtask_sequence[idx] = [actions[pt1], 0]
                    pt1 += 1
                elif pt2 < len(atom_objects_idxes) and idx == atom_objects_idxes[pt2]:
                    parsed_subtask_sequence[idx] = [atom_objects[pt2], 1]
                    pt2 += 1

            parsed_subtask_sequences.append(parsed_subtask_sequence)

        return parsed_subtask_sequences

    def navigation_parsing_v2(self, navigation_task):
        parsed_subtask_sequences = []

        sub_tasks = self.coarse_task_parsing(navigation_task)

        # subtask corpus to commands and objects
        parsed_subtask_corpuses = []
        for sub_task in sub_tasks:
            time.sleep(3)
            parsed_subtask_corpus = self.sub_task_parsing_v2(sub_task)
            parsed_subtask_corpuses.append(parsed_subtask_corpus)

        parsed_subtasks = []
        for parsed_subtask_corpus in parsed_subtask_corpuses:
            atom_corpuses = [t for t in parsed_subtask_corpus.split(";") if t]

            parsed_atom_tasks = []
            for idx, atom_corpus in enumerate(atom_corpuses):

                atom_corpus = atom_corpus.strip("\n").strip(" ")
                # matches = re.findall(r'<(.*?)>.*?<(.*?)>', atom_corpus)[0]
                print("atom_corpus: ", atom_corpus)
                # action_corpus, target = re.findall(r'.*?<(.*?)>.*?<(.*?)>.*?', atom_corpus)[0]
                matches = re.findall(r'<([^>]+)>', atom_corpus)
                print(matches)
                if len(matches) == 0:
                    continue
                elif len(matches) == 1:
                    action_corpus = matches[0]
                    target = ""
                else:
                    action_corpus, target = matches
                actions = action_corpus.split(",")

                for i, act in enumerate(actions):
                    act = act.strip(" ").strip(",")
                    if "TAKE OFF" in act:
                        if idx == 0 and i == 0:
                            parsed_atom_tasks.append((act, 0))
                    else:
                        parsed_atom_tasks.append((act, 0))

                if "TAKE OFF" not in actions[0] or len(actions) != 1:       # exclude case: <TAKE OFF, target>
                    if target != "":
                        parsed_atom_tasks.append((target, 1))

            parsed_subtasks.append(parsed_atom_tasks)

        return parsed_subtasks

    def coarse_task_parsing(self, task):
        task = task.strip(" ")
        sub_tasks = [t.strip(" ") for t in task.split(".") if t]

        return sub_tasks

    def sub_task_parsing(self, sub_task, show_response=False):
        prompt = f"""
        Given a navigation task: '{sub_task}', please extract mentioned actions and objects in order. 
        The object should be concrete and not be pronoun.
        To distinguish task and object, we label action as 0 and object as 1.   
        Based on such a provision, your output should be such format:
            <action, label>;<action, label>;<object, label>;<action, label>;...
        
        Example output:
        take off,0;turn left,0;the building,1;street light,1;turn right,0
        """

        response = self.llm.query_api(prompt, show_response=show_response)

        return response

    def sub_task_parsing_v2(self, sub_task, show_response=False):
        prompt = f"""
        Given a UAV navigation task: '{sub_task}', please break down the navigation task into atom tasks. 
        The atom tasks should be formatted as <ACTION, TARGET>. Each atom task is composed of a sequence of actions and a target.
        The action should ONLY be selected from ["TAKE OFF", "GO UP", "GO DOWN", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "TOUCH DOWN"].
        The target should be concrete object.
        Example output:
        <ACTION1, ACTION2, ACTION3,...>,<TARGET>;<ACTION1, ACTION2, ACTION3,...>,<TARGET>;...
        """

        response = self.llm.query_api(prompt, show_response=show_response)

        return response

    def action_mapping(self, control_instructions):
        action_space = [act for act in DefaultAirsimActionCodes]
        prompt = f"""
        You are an AI agent helping to control the drone to finish the navigation task.
        You have a list of control command, you need to map EACH instruction to ONE action in your action space.
        Your action space is {action_space}.
        Your control commands are {control_instructions}.
        
        Your output format should ONLY be:
        <action>;<action>;<action>;...
        
        Example:
        GO UP;TURN LEFT;MOVE FORWARD;GO DOWN;STOP
        """
        response = self.llm.query_api(prompt, show_response=False)
        actions = response.strip(" ").split(";")
        return actions

    def checkpoint_parsing(self, navi_instruction):
        navi_instruction = "take off and turn left fly over the trees. now turn left and cross over the pond and turn right and move towards the football court ground. now slow down to the floor and stay there."
        subtasks = navi_instruction.split(".")
        subtask_checkpoints = [[] for _ in range(len(subtasks))]
        subtask_checkpoints[0] = ["take off", "turn left", "fly over the trees"]
        subtask_checkpoints[1] = ["turn left", "cross over the pond", "turn right", "move towards the football court ground"]
        subtask_checkpoints[2] = ["slow down to the floor", "stay there"]

        return subtasks, subtask_checkpoints


if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    navigation_task = "take off and turn left fly over the trees. now turn left and cross over the pond and turn right and move towards the football court ground. now slow down to the floor and stay there."

    navi_task_file = "/home/vincent/py-pro/AirVLN-main/DATA/data/aerialvln/processed_val_seen.json"

    with open(navi_task_file) as f:
        navi_data_raw = json.load(f)

    navi_data_raw = navi_data_raw["episodes"]
    for i in range(5):
        if i >= 2:
            break

        print(navi_data_raw[i]["instruction_text"])
        navi_instruction = navi_data_raw[i]["instruction_text"]

        llm = OpenAI_LLM(
            max_tokens=4096,
            model_name="gpt-4",
            api_key=OPENAI_API_KEYS,
            cache_name="navigation",
            finish_reasons=["stop", "length"],
        )

        parser = TaskParser(llm)
        # subtasks = parser.task_parsing(navi_instruction)
        #
        # for subtask in subtasks:
        #     parsed_subtask = parser.sub_task_parsing(subtask,show_response=False)
        #     time.sleep(5)
        #     print(subtask)
        #     print(parsed_subtask)

        parsed_subtask_sequences = parser.navigation_parsing_v2(navi_instruction)

        print(parsed_subtask_sequences)
        for parsed_subtask_sequence in parsed_subtask_sequences:
            print(parsed_subtask_sequence)


