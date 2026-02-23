import torch
import random
from typing import List
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
from alfworld.agents.utils.misc import get_templated_task_desc
from alf_utils import AlfEnv

def get_alfworld_prompt(env_name, action_history, admissible_actions, action_only = False):
    """
        This function defines the prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    task = get_templated_task_desc(env_name.env.envs[0].traj_data)
    if not action_only:
        refomratted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions)
        qs = f"Your are an expert in the ALFRED Embodied Environment."
        qs = qs + f"Your task is to " + task + ". "
        qs = qs + f"You are also given the previous actions you have taken: {action_history}. "
        qs = qs + f"Your admissible actions of the current situation are: [{refomratted_admissible_actions}]. "
        qs = qs + "Your response should be a valid json file in the following format: \n\{\n"
        qs = qs + "\"thoughts\": \"{first describe what do you see in the image using the text description, then carefully think about which action to complete the task based on your observation, action history and admissible actions. }\", \n"
        qs = qs + "\"action\": \"{an admissible action}\"\n\}"
    else:
        refomratted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions)
        qs = f"Your are an expert in the ALFRED Embodied Environment."
        qs = qs + f"Your task is to " + task + ". "
        qs = qs + f"You are also given the previous actions you have taken: {action_history}. "
        qs = qs + f"Your admissible actions of the current situation are: [{refomratted_admissible_actions}]. "
        qs = qs + "Your response should be a valid json file in the following format: \n\{\n"
        qs = qs + "\"action\": \"{an admissible action}\"\n\}"
    return qs, task