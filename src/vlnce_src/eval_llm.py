import os
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(str(os.getcwd())).resolve()))
import gc
import time
import lmdb
import tqdm
import math
import random
import json
import airsim
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from typing import List, Optional, DefaultDict
import msgpack_numpy

from utils.logger import logger
from utils.utils import get_rank, is_dist_avail_and_initialized, is_main_process, init_distributed_mode
from utils.control_utils import action_str2enum
from Model.il_trainer import VLNCETrainer
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import observations_to_image, append_text_to_image, generate_video

from src.common.param import args
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import read_vocab, Tokenizer

from src.llm.prompt_builder import (
    visual_observation_prompt_builder,
    open_ended_action_manager_prompt_builder_v2,
    summarize_view_observation,
    relative_spatial_prompt_builder,
    cot_prompt_builder_p1,
    cot_prompt_builder_p2,
    cot_prompt_builder_p3,
    subtask_action_manager_prompt_builder,
)

from src.llm.query_llm import OpenAI_LLM, Qwen_VL
from secret.keys import OPENAI_API_KEYS, DASHSCOPE_API_KEY

from airsim_plugin.airsim_settings import ObservationDirections, DefaultAirsimActionNames

inverse_direction = {
    "north": "south",
    "south": "north",
    "west": "east",
    "east": "west"
}

def setup():

    init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    # max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    max_traj_len = 500
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid][:max_traj_len, ...], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid][:max_traj_len, ...], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


@torch.no_grad()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def initialize_tokenizer():
    if args.tokenizer_use_bert:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        vocab = read_vocab(args.TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    return tok


def initialize_env(split='train'):
    tok = initialize_tokenizer()

    train_env = AirVLNENV(batch_size=args.batchSize, split=split, tokenizer=tok)

    return train_env


def initialize_trainer():
    from gym import spaces
    from airsim_plugin.airsim_settings import AirsimActions

    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
        "instruction": spaces.Discrete(0),
        "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
    })
    action_space = spaces.Discrete(int(len(AirsimActions)))

    trainer = VLNCETrainer(
        load_from_ckpt=False,
        observation_space=observation_space,
        action_space=action_space,
    )

    logger.info('initialize_trainer over')
    return trainer


def eval_vlnce():
    logger.info(args)

    writer = TensorboardWriter(
        str(Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        flush_secs=30,
    )

    tok = initialize_tokenizer()
    logger.info("token initialize success!")

    assert os.path.exists(args.EVAL_CKPT_PATH_DIR), '评估文件(夹)不存在'

    _eval_llm_navigator(writer, tok)

    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)
    logger.info("END evaluate")


def _eval_llm_navigator(
        writer=None,
        tok=None,
        checkpoint_index: int = 0
) -> None:
    logger.info(f"start navigation")

    if args.EVAL_DATASET == 'train':
        train_env = AirVLNENV(batch_size=args.batchSize, split='train', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_unseen', tokenizer=tok)
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNENV(batch_size=args.batchSize, split='test', tokenizer=tok)
    else:
        raise KeyError

    # loda llm
    llm = OpenAI_LLM(
        max_tokens=4096,
        model_name="gpt-4-vision-preview",
        api_key=OPENAI_API_KEYS,
        cache_name="navigation",
        finish_reasons=["stop", "length"],
    )

    vln = Qwen_VL(
        model_name="qwen-vl-max",
        api_key=DASHSCOPE_API_KEY,
        cache_name="navigation",
        max_tokens=4096,
    )

    #
    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return

    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    start_iter = 0
    end_iter = len(train_env.data)
    cnt = 0
    success_cnt = 0
    print("total case: ", end_iter)
    for idx in range(start_iter, end_iter, train_env.batch_size):
        if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
            break
        cnt += 1

        print("\nCurrent count: {}\n".format(cnt))

        train_env.next_minibatch(skip_scenes=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
        if train_env.batch is None:
            logger.warning('train_env.batch is None, going to break and stop collect')
            break

        prev_action = "None"

        outputs = train_env.reset()
        observations, _, dones, _ = [list(x) for x in zip(*outputs)]
        batch = batch_obs(observations) # batched training data, dict, each item in the dict is a batched tensor

        # infos, rgb_frames = naive_navigation_prompter(train_env, llm, vln, pbar)
        # infos, rgb_frames = COT_navigation_prompter(train_env, llm, vln, pbar)
        infos, rgb_frames = discrete_navigation_prompter(train_env, llm, vln, pbar)
        if infos is not None:
            for t in range(int(train_env.batch_size)):
                logger.info((
                        'result-{} \t' +
                        'distance_to_goal: {} \t' +
                        'success: {} \t' +
                        'ndtw: {} \t' +
                        'sdtw: {} \t' +
                        'path_length: {} \t' +
                        'oracle_success: {} \t' +
                        'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['success'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                ))
                if infos[t]['success'] > 0.5:
                    success_cnt += 1

        print(
            "\ntotal task count: {}, success count: {}, success rate: {}\n".format(cnt * train_env.batch_size, success_cnt,
                                                                               success_cnt * 1.0 / (
                                                                                           cnt * train_env.batch_size)))
        if args.EVAL_GENERATE_VIDEO:
            h, w = rgb_frames[0][0].shape[:2]
            print("frames: ", len(rgb_frames[0]))

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            # Create a VideoWriter object. Arguments are:
            # 1. The output file name (eg: output.avi)
            # 2. FourCC code
            # 3. Frames per second
            # 4. Frame size (width, height) as a tuple
            out = cv2.VideoWriter('files/videos/output_{}.avi'.format(cnt), fourcc, 1, (w, h))

            # Assume 'frames' is a list of numpy arrays containing your frames
            # For demonstration, let's create 60 frames with random content

            for frame in rgb_frames[0]:
                # Writing each frame into the video file
                # Make sure the frame is in BGR format if using OpenCV
                out.write(frame)

            # Release the VideoWriter object
            out.release()

            print("Video processing complete.")
        # end
        pbar.close()
    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


def naive_navigation_prompter(sim_env, llm, vln, pbar):
    logger.info("Start navigation")

    infos = None
    rgb_frames = [[] for _ in range(sim_env.batch_size)]

    skips = [False for _ in range(sim_env.batch_size)]

    batch_raw = sim_env.batch  # a batch of dict, each dict is groundtruth
    navi_gt = batch_raw[0]

    history_actions = []
    sim_env.makeActions([4], update_statue=False)  # take off

    for t in range(int(args.maxAction)):
        # state
        drone_state = sim_env.sim_states[0]

        # navigation
        navi_instruct = navi_gt["instruction"]["instruction_text"]
        print("Navigation instruction: {}".format(navi_instruct))
        time.sleep(10)

        # get pano observations
        action_list = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        direction_idx = 0
        obs_list = []
        for sub_action in action_list:
            obs_per_img_list = []
            obs = sim_env.get_obs()
            rgb_frame = obs[0][0]["rgb"]
            img_p = "/home/vincent/py-pro/AirVLN-main/AirVLN/files/temp_rgb.png"

            cv2.imwrite(img_p, rgb_frame)
            time.sleep(5)
            cleaned_grounding, response = vln.query_api("Generate the caption in English with grounding: ", img_p,
                                                        show_response=False)
            # print(
            #     "current {} image observation: {}".format(
            #         observed_direction[direction_idx], cleaned_grounding
            #     )
            # )
            # print(response)

            for o in cleaned_grounding:
                obs_per_img_list.append(o["object"])
            print("obs_per_img_list: ", obs_per_img_list)
            direction_idx += 1
            obs_list.append(obs_per_img_list)

            # rotate by yaw
            for act in sub_action:
                sim_env.makeActions([act], update_statue=False)

        visual_observation_summary = summarize_view_observation(obs_list)
        action_prompt = open_ended_action_manager_prompt_builder_v2(
            navi_instruct,
            visual_observation_summary,
            history_actions
        )

        # print("action prompt: ", action_prompt)
        time.sleep(10)
        response = llm.query_api(action_prompt, show_response=False)
        print("action response: ", response)

        action_str = response.split(":")[0]
        reasons = " ".join(response.split(":")[1:])
        # print(action_str)
        print(reasons)
        prev_action = action_str
        action = action_str2enum(action_str)
        sim_env.makeActions([action])

        history_actions.append(action_str)
        outputs = sim_env.get_obs()
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]

        print("Trajectory: ", drone_state.trajectory)
        print("History actions: ", history_actions)
        for i in range(sim_env.batch_size):
            if args.EVAL_GENERATE_VIDEO:
                frame = observations_to_image(observations[i], infos[i])
                frame = append_text_to_image(
                    frame, sim_env.batch[i]['instruction']['instruction_text']
                )
                rgb_frames[i].append(frame)

            if not dones[i] or skips[i]:
                continue

            skips[i] = True
            pbar.update()

        if np.array(dones).all():
            ended = True
            break

    return infos, rgb_frames


def COT_navigation_prompter(sim_env, llm, vln, pbar):
    logger.info("Start navigation")

    infos = None
    rgb_frames = [[] for _ in range(sim_env.batch_size)]

    skips = [False for _ in range(sim_env.batch_size)]

    batch_raw = sim_env.batch  # a batch of dict, each dict is groundtruth
    navi_gt = batch_raw[0]

    history_actions = []
    current_subgoal = "take off"
    sim_env.makeActions([4])  # take off

    # environment warm up
    for _ in range(5):
        _ = sim_env.get_obs("front_0")
        _ = sim_env.get_obs("bottom_0")

    # start navigation
    for t in range(int(args.maxAction)):

        # state
        drone_state = sim_env.sim_states[0]

        # navigation
        navi_instruct = navi_gt["instruction"]["instruction_text"]
        print("Navigation instruction: {}".format(navi_instruct))

        # get pano observations
        # action_list = [[2,2], [2,2], [2,2], [2,2]]
        obs_list = [[] for _ in range(len(ObservationDirections))]
        # assert len(action_list)+1 == len(ObservationDirections)
        for obs_idx in range(len(ObservationDirections)):
            if ObservationDirections[obs_idx] == "BOTTOM":
                obs = sim_env.get_obs(camera_id="bottom_0")
            else:
                obs = sim_env.get_obs(camera_id="front_0")

            rgb_frame = obs[0][0]["rgb"]
            img_p = "/home/vincent/py-pro/AirVLN-main/AirVLN/files/rgb_obs_{}.png".format(ObservationDirections[obs_idx])

            cv2.imwrite(img_p, rgb_frame)
            time.sleep(5)
            cleaned_grounding, response = vln.query_api("Generate the caption in English with grounding: ", img_p,
                                                        show_response=False)

            for o in cleaned_grounding:
                obs_list[obs_idx].append(o["object"])
            print("{} observation: {}".format(ObservationDirections[obs_idx], obs_list[obs_idx]))

            # rotate by yaw
            if obs_idx < len(ObservationDirections)-1:
                sim_env.makeVirtualActions([[2, 2,2,2,2,2]])

        visual_observation_summary = summarize_view_observation(obs_list)

        # cot_prompt = cot_prompt_builder_p1(navi_instruct, history_actions)
        # current_subgoal = llm.query_api(cot_prompt, show_response=False)
        # print("p1_response: ", current_subgoal)
        # time.sleep(5)

        current_position = sim_env.sim_states[0].pose.position
        cot_prompt = cot_prompt_builder_p2(navi_instruct, history_actions, current_subgoal, visual_observation_summary, current_position)
        subgoal_res = llm.query_api(cot_prompt, show_response=False)
        subgoal_status = subgoal_res.split(":")[0]
        subgoal_reason = "".join(subgoal_res.split(":")[1:])
        print("p2_response: ", subgoal_res)
        time.sleep(5)

        # update subgoal
        if "YES" in subgoal_status:
            cot_prompt      = cot_prompt_builder_p3(navi_instruct, history_actions, current_subgoal, True, visual_observation_summary)
            current_subgoal = llm.query_api(cot_prompt, show_response=False)
            time.sleep(5)

        cot_prompt      = cot_prompt_builder_p3(navi_instruct, history_actions, current_subgoal, False, visual_observation_summary)
        action_res      = llm.query_api(cot_prompt, show_response=False)
        time.sleep(5)

        print("action_response: ", action_res)
        action_str = action_res.split(":")[0]
        reasons = " ".join(action_res.split(":")[1:])

        action = action_str2enum(action_str)
        sim_env.makeActions([action])

        history_actions.append(action_str)
        outputs = sim_env.get_obs()
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]

        print("Trajectory: ", drone_state.trajectory)
        print("History actions: ", history_actions)
        for i in range(sim_env.batch_size):
            if args.EVAL_GENERATE_VIDEO:
                frame = observations_to_image(observations[i], infos[i])
                frame = append_text_to_image(
                    # frame, sim_env.batch[i]['instruction']['instruction_text']
                    frame, sim_env.batch[i]['instruction']['instruction_text']+"\n"+subgoal_res+"\n"+action_res
                )
                rgb_frames[i].append(frame)

            if not dones[i] or skips[i]:
                continue

            skips[i] = True
            pbar.update()

        if np.array(dones).all():
            ended = True
            break

        input("press any key to continue.")

    return infos, rgb_frames


def discrete_navigation_prompter(sim_env, llm, vln, pbar):
    logger.info("Start navigation")

    infos = None
    rgb_frames = [[] for _ in range(sim_env.batch_size)]

    batch_raw = sim_env.batch  # a batch of dict, each dict is groundtruth
    navi_gt = batch_raw[0]
    subtasks, subtask_checkpoints, subtask_checkpoint_node_poses, subtask_checkpoint_node_actions \
        = sim_env.navi_task_preprocessing(navi_gt)

    # [-345.41436684853386, -62.60882195838579, -25.112414836883545, 0, 0, -1.8705922177616467]
    # reset pose to the initial
    initial_pose = [-345.41436684853386, -62.60882195838579, -5.112414836883545, 0, 0, -1.6087928299624972] # navi_gt["reference_path"][0]
    initial_pos = initial_pose[:3]
    initial_rot = airsim.to_quaternion(initial_pose[3], initial_pose[4], initial_pose[5])

    initial_pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=initial_pos[0],
                        y_val=initial_pos[1],
                        z_val=initial_pos[2],
                    ),
                    orientation_val=initial_rot,
                )
    sim_env.reset_to_this_pose([[initial_pose]], need_change=False)
    sim_env.sim_states[0].pose = initial_pose
    print(navi_gt["instruction"]["instruction_text"])
    # environment warm up
    for _ in range(5):
        _ = sim_env.get_obs("front_0")
        _ = sim_env.get_obs("bottom_0")

    # start navigation
    ckt_pt = 0
    subtask_pt = 1

    finished_subtasks = []
    finished_checkpoints = []
    cnt = 0
    while cnt < args.maxAction:
        ongoing_subtask = subtasks[subtask_pt]
        ongoing_checkpoint = subtask_checkpoints[subtask_pt][ckt_pt]
        node_pt = 0
        step_from_prev_ckt = 0
        init_ckt_pose = sim_env.sim_states[0].pose
        on_track = True

        # finish checkpoint
        while True:
            obs_list = []
            obs_img_paths = []
            print("drone state: {}".format(sim_env.sim_states[0].pose))

            for obs_idx in range(len(ObservationDirections)):
                if ObservationDirections[obs_idx] == "BOTTOM":
                    # todo: bug: bottom 0 image will occur safety issue of the GPT4
                    if cnt == 0:
                        obs = sim_env.get_obs(camera_id="front_0")
                    else:
                        obs = sim_env.get_obs(camera_id="bottom_0")
                else:
                    obs = sim_env.get_obs(camera_id="front_0")


                rgb_frame = obs[0][0]["rgb"]
                img_p = "/home/vincent/py-pro/AirVLN-main/AirVLN/files/rgb_obs_{}.png".format(
                    ObservationDirections[obs_idx])

                cv2.imwrite(img_p, rgb_frame)
                obs_img_paths.append(img_p)
                # time.sleep(5)
                visual_observation_prompt = visual_observation_prompt_builder()
                # response = llm.query_api(visual_observation_prompt, img_p, show_response=False)

                # obs = [t.strip(" ") for t in response.split(";")]
                # obs_list.append(obs)

                # rotate by yaw
                if obs_idx < len(ObservationDirections) - 1:
                    sim_env.makeVirtualActions([[2, 2, 2, 2, 2, 2]])

            print(obs_img_paths)
            response = llm.query_apis("from left to right, summary objects in each image, the index is started by 0. your output should be formatted as: '0. {objects}, 1. {objects}, ...'", obs_img_paths)

            visual_observation_summary = summarize_view_observation(obs_list)

            subtask_action_prompt = subtask_action_manager_prompt_builder(ongoing_subtask, finished_checkpoints, ongoing_checkpoint, visual_observation_summary)

            time.sleep(3)
            action_res = llm.query_api(subtask_action_prompt, show_response=False)

            print("action_response: ", action_res)
            action_str = action_res.split(":")[0]
            reasons = " ".join(action_res.split(":")[1:])

            action = action_str2enum(action_str)
            if action == subtask_checkpoint_node_actions[subtask_pt][ckt_pt]["action_code"] and on_track:
                node_actions = subtask_checkpoint_node_actions[subtask_pt][ckt_pt]["actions"][node_pt]
                for act in node_actions:
                    sim_env.makeActions([act])
                    t = sim_env.sim_states[0].pose
                node_pt += 1
                cnt += len(node_actions)
                step_from_prev_ckt += 1
                print("On the track. Action: {}. Current node pointer: {}. Current checkpoint: {}, step from previous checkpoint: {}".
                      format(action_str, node_pt, subtask_checkpoints[subtask_pt][ckt_pt], step_from_prev_ckt))
            else:
                sim_env.makeActions([action])
                cnt += 1
                step_from_prev_ckt += 1
                on_track = False
                print("Off the track. Action: {}. Current node pointer: {}. Current checkpoint: {}. step from previous checkpoint: {}".
                      format(action_str, node_pt, subtask_checkpoints[subtask_pt][ckt_pt], step_from_prev_ckt))

            if args.EVAL_GENERATE_VIDEO:
                outputs = sim_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                for i in range(sim_env.batch_size):
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        # frame, sim_env.batch[i]['instruction']['instruction_text']
                        frame,
                        sim_env.batch[i]['instruction']['instruction_text'] + action_res
                    )
                    rgb_frames[i].append(frame)

                    pbar.update()


            # if current checkpoint is finished or exceeded max action
            if node_pt == len(subtask_checkpoint_node_actions[subtask_pt][ckt_pt]["actions"]) or cnt >= args.maxAction:
                break

            # if not on track for long, reset to the initial pose
            if step_from_prev_ckt >= 20:
                on_track = True
                step_from_prev_ckt = 0
                node_pt = 0
                sim_env.reset_to_this_pose([[init_ckt_pose]], need_change=False)
                sim_env.sim_states[0].pose = initial_pose
                print("Step from previous checkpoint threshold reached, reset to previous checkpoint.")

        print("current checkpoint {} finished".format(subtask_checkpoints[subtask_pt][ckt_pt]))
        finished_checkpoints.append(subtask_checkpoints[subtask_pt][ckt_pt])
        ckt_pt += 1

        if ckt_pt == len(subtask_checkpoints[subtask_pt]):
            finished_subtasks.append(subtasks[subtask_pt])
            subtask_pt += 1

            ckt_pt = 0
            finished_checkpoints = []

        if subtask_pt == len(subtasks):
            print("Navigation task finished")
            break

    return infos, rgb_frames

def visualize_gt():
    gt_path = "/home/vincent/py-pro/AirVLN-main/DATA/data/aerialvln/val_seen.json"
    with open(gt_path) as f:
        navi_data_raw = json.load(f)

    tok = initialize_tokenizer()
    train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)

    cnt = 0
    for idx in range(0, len(train_env.data), train_env.batch_size):
        if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
            break
        cnt += 1
        rgb_frames = [[] for _ in range(train_env.batch_size)]
        print("\nCurrent count: {}\n".format(cnt))

        train_env.next_minibatch(skip_scenes=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
        if train_env.batch is None:
            print('train_env.batch is None, going to break and stop collect')
            break

        outputs = train_env.reset()
        c_data = train_env.batch[0]
        print(c_data["trajectory_id"])
        actions = c_data["actions"]
        print(actions)
        print(len(actions))
        print(len(c_data["reference_path"]))

        print(actions[144:150], actions[85])

        # a, b, _, d = train_env.navi_task_preprocessing(c_data)
        # print(a, b, d)
        # break
        for i, act in enumerate(actions):
            outputs = train_env.get_obs(camera_id="front_0")
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            for j in range(train_env.batch_size):
                if args.EVAL_GENERATE_VIDEO:
                    frame = observations_to_image(observations[j], infos[j])
                    frame = append_text_to_image(
                        # frame, sim_env.batch[i]['instruction']['instruction_text']
                        frame,
                        train_env.batch[j]['instruction']['instruction_text']+DefaultAirsimActionCodes[act]
                    )
                    # print(frame.shape)
                    rgb_frames[0].append(frame)

            train_env.makeActions([act])
            # print(DefaultAirsimActionNames[act])

        if args.EVAL_GENERATE_VIDEO:
            h, w = rgb_frames[0][0].shape[:2]
            print("frames: ", len(rgb_frames[0]))

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            out = cv2.VideoWriter('files/videos/{}.avi'.format(c_data["trajectory_id"]), fourcc, 2, (w, h))
            for frame in rgb_frames[0]:
                out.write(frame)

            # Release the VideoWriter object
            out.release()

            print("Video processing complete.")

        if cnt >=2:
            break
    # navi_data_raw = navi_data_raw["episodes"]
    # print(navi_data_raw[0])
    # # for i, navi_traj in enumerate(navi_data_raw):
    # reference_path = navi_data_raw[0]["reference_path"]
    # reference_action = navi_data_raw[0]["actions"]
    # print(len(reference_action))


if __name__ == "__main__":
    setup()
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    eval_vlnce()
    visualize_gt()
