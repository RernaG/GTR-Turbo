import time
from collections import deque
import os
import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_alfworld_prompt
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage_sft import RolloutStorageSFT
from a2c_ppo_acktr.storage_kl import RolloutStorageKL

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from a2c_ppo_acktr.qwen_interface import find_target_linear_names, obtain_prompt_text
from a2c_ppo_acktr.qwen_interface import qwen_evaluate, qwen_generate

# For alfworld
from alf_utils import AlfEnv

from image_logger import ImageLogger

from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import datetime

import warnings
warnings.filterwarnings("ignore")

from guider import MergedModelGuider
from metric import get_token_length, get_self_bleu

def main():
    args = get_args()

    assert torch.cuda.device_count() >= 2
    assert args.tht_guide == 'SFT' or args.tht_guide == 'KL'
    is_sft_guide = (args.tht_guide == 'SFT')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    model_device = device
    model_path = args.model_path
    cache_dir = args.cache_dir

    print(f"Path of the model is {model_path}")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=cache_dir,
    )
    image_processor = AutoProcessor.from_pretrained(
        model_path,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
    )

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_target_linear_names(base, lora_namespan_exclude=['lm_head', 'embed_tokens'], num_lora_modules=-1),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)

    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    ## Inputing Prompt here
    assert args.alf_config is not None, "Alfworld environment requires a config file"
    envs = AlfEnv(args.alf_config)

    # hardcoding guider in GPU #1
    guider = MergedModelGuider(model_path, 1)

    obs, infos = envs.reset(seed=args.seed)
    # text_ob, infos, done, obs = envs.send_command("RESET")
    admissible_commands = list(infos['admissible_commands'])[0]
    qs, task = get_alfworld_prompt(envs, action_history = [], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
    PROMPT_TEXT = obtain_prompt_text(image_processor, qs)
    print(PROMPT_TEXT)

    if "alfred" in args.env_name.lower():
        projection_f = partial(lambda x: x)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             PROMPT_TEXT=PROMPT_TEXT,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm)

    if is_sft_guide:
        rollouts = RolloutStorageSFT(args.num_env_steps, args.num_steps, args.num_processes, (300, 300, 3), spaces.Discrete(14), args.max_new_tokens)
    else:
        rollouts = RolloutStorageKL(args.num_env_steps, args.num_steps, args.num_processes, (300, 300, 3), spaces.Discrete(14), args.max_new_tokens)

    _, _, output_ids, action, action_log_prob, action_tokens_log_prob, _ = actor_critic.act(obs, PROMPT_TEXT=PROMPT_TEXT)
    admissible_commands = list(infos['admissible_commands'])[0]

    print("output_ids:{}".format(output_ids))
    print("prompt:{}".format(PROMPT_TEXT))
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    if is_sft_guide:
        rollouts.obs_accu[0].copy_(obs)
    rollouts.to('cpu')
    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_gc_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_and_place = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_two_obj_and_place = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_look_at_obj_in_light = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_heat_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_cool_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_clean_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)
    output_tolen = []
    action_history = []
    valid_prob = []
    thought_tolen = []
    episode_lengths = []
    answer_history = []

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    if args.use_wandb:
        import wandb
        run_name = args.tag
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)
    
    if args.step_log and accelerator.is_main_process:
        step_log_path = os.path.join(args.steplogdir, args.tag)
        os.makedirs(step_log_path, exist_ok=True)
        step_logger = ImageLogger(step_log_path)

    # print("prompt:{}".format(PROMPT_TEXT))
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    for j in tqdm(range(num_updates)):
        output_tolen.clear()
        thought_tolen.clear()
        valid_prob.clear()
        episode_lengths.clear()
        episode_len = 0
        for step in range(args.num_steps):
            print(f"-------- UPDATE {j+1}, STEP {step+1} --------")

            print(f"ACTION HISTORY: {action_history}")
            if "alfred" in args.env_name.lower():
                admissible_commands = list(infos['admissible_commands'])[0]
                qs, task = get_alfworld_prompt(envs, action_history=action_history, admissible_actions=admissible_commands, action_only = args.action_only_prompt)
                PROMPT_TEXT = obtain_prompt_text(image_processor, qs)

            # Sample actions
            with torch.no_grad():
                value, INPUT_IDS, output_id, action, action_log_prob, action_tokens_log_prob, raw_output_id = actor_critic.act(
                        rollouts.obs[step], PROMPT_TEXT = PROMPT_TEXT)
                admissible_commands = list(infos['admissible_commands'])[0]
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text_action = text_action.strip().removeprefix("```json").removesuffix("```").strip()
            answer_history.append(text_action)
            print(f"TEXT ACTION: {text_action}")
            tolen = get_token_length(tokenizer, text_action)
            output_tolen.append(tolen)

            img = rollouts.obs[step][0].cpu().numpy().astype(np.uint8)
            if is_sft_guide:
                correct_ids, labels, format_reward, valid, thought_token_length, guider_answer = guider.thought_guide_sft(PROMPT_TEXT, text_action, img)
                correct_ids = correct_ids.to(model_device)
                input_ids = torch.cat([INPUT_IDS, correct_ids], dim=1).to(model_device)
                padded_input_ids = torch.zeros(input_ids.size(0), 1500).to(dtype=input_ids.dtype, device = input_ids.device)
                padded_input_ids[:, :input_ids.size(1)] = input_ids
                
                padded_labels = torch.full((1, 1500), -100)
                padded_labels[:, INPUT_IDS.size(1): input_ids.size(1)] = labels
                padded_labels = padded_labels.to(input_ids.device)
            else:
                format_reward, valid, thought_token_length, kl_loss = guider.thought_guide_kl(PROMPT_TEXT, text_action, img, actor_critic.base, args)
            valid_prob.append(valid)
            if thought_token_length != None:
                thought_tolen.append(thought_token_length)

            prev_infos = copy.deepcopy(infos)
            # Observation, reward and next obs
            obs, reward, done, infos = envs.step(action) # for alf this will already process action
            if args.step_log:
                ref_answer = guider_answer if is_sft_guide else ""
                step_logger.step_log(j+1, step+1, rollouts.obs[step][0].cpu().numpy().astype(np.uint8), text_action, prev_infos, task, reward.item(), action_history, ref_answer)
            action_history.append(action[0])
            episode_len += 1

            print(f"INFOS: {infos}")
            print(f"REWARD: {reward.item()}")

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            if len(action_history) > 40 or (len(action_history) >= 5 and all(x == action_history[-1] for x in action_history[-5:])):
                print("TRUNCATED!")
                done = [True]
                reward = torch.FloatTensor([-3.0])

            self_bleu_score = 0
            if done[0]:
                if len(answer_history) > 1:
                    self_bleu_score = get_self_bleu(answer_history)
                answer_history.clear()
                print("SELF_BLEU", self_bleu_score)

            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    # record success rate of different types of tasks
                    if "pick_and_place" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_and_place.append(float(infos['won'][0]))
                    elif "pick_two_obj_and_place" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_two_obj_and_place.append(float(infos['won'][0]))
                    elif "look_at_obj_in_light" in infos["extra.gamefile"][0]:
                        episode_succ_rate_look_at_obj_in_light.append(float(infos['won'][0]))
                    elif "pick_heat_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_heat_then_place_in_recep.append(float(infos['won'][0]))
                    elif "pick_cool_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_cool_then_place_in_recep.append(float(infos['won'][0]))
                    elif "pick_clean_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_clean_then_place_in_recep.append(float(infos['won'][0]))
                    # record the final success rate
                    episode_success_rate.append(float(infos['won'][0]))
                    episode_gc_success_rate.append(float(infos['goal_condition_success_rate'][0]))
                    print(len(episode_success_rate))
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
                    obs, infos = envs.reset()
                    action_history = []
                    episode_lengths.append(episode_len)
                    episode_len = 0

            # bad_masks is a legact implementation in the storage
            bad_masks = torch.zeros(args.num_processes, 1)
            # action_id is also a legacy implementation in the storage, it is never used in the PPO update
            action_id = None
            for i in range(len(admissible_commands)):
                if admissible_commands[i] == action:
                    action_id = i
                    break
            if not action_id:
                action_id = 0
            action_id = torch.tensor(action_id)

            if is_sft_guide:
                rollouts.insert(obs, output_id, action_id,
                                action_log_prob, value, reward, masks, bad_masks, format_reward, padded_input_ids, padded_labels, self_bleu_score)
            else:
                rollouts.insert(obs, output_id, action_id,
                                action_log_prob, value, reward, masks, bad_masks, format_reward, self_bleu_score, kl_loss)

        # print("prompt:{}".format(PROMPT_TEXT))
        # print("action_log_prob:{}".format(action_log_prob))
        # print("text_action:{}".format(text_action))
        # print("action:{}".format(action))
        # print("ground truth:{}".format(infos))
        print("success_rate:{}".format(np.mean(episode_success_rate)))

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], PROMPT_TEXT = PROMPT_TEXT).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy, ppo_loss, sft_loss = agent.update(rollouts, j, args)
        lr_scheduler.step()
        guider.merge(args.output_dir, args.tag, j, args.use_ema, args.ema_alpha)

        rollouts.after_update()
        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}, dist_entropy {:.4f}, value_loss {:.4f}, action_loss {:.4f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate),
                        dist_entropy, value_loss, action_loss))
            if args.use_wandb:
                # wandb_images = [wandb.Image(image.cpu().numpy()) for image in obs]
                # text_table.add_data(j, infos['observation_text'][0], text_action)
                wandb.log({"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "episode/reward.mean": np.mean(episode_rewards),
                        "episode/reward.median": np.median(episode_rewards),
                        "episode/reward.min": np.min(episode_rewards),
                        "episode/reward.max": np.max(episode_rewards),
                        "episode/success_rate.mean": np.mean(episode_success_rate),
                        "episode/action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                        "episode/(goal_condition)_success_rate.mean": np.mean(episode_gc_success_rate),
                        "episode/succ_rate_pick_and_place.mean": np.mean(episode_succ_rate_pick_and_place),
                        "episode/succ_rate_pick_two_obj_and_place.mean": np.mean(episode_succ_rate_pick_two_obj_and_place),
                        "episode/succ_rate_look_at_obj_in_light.mean": np.mean(episode_succ_rate_look_at_obj_in_light),
                        "episode/succ_rate_pick_heat_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_heat_then_place_in_recep),
                        "episode/succ_rate_pick_cool_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_cool_then_place_in_recep),
                        "episode/succ_rate_pick_clean_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_clean_then_place_in_recep),
                        "episode/num": len(episode_success_rate),
                        "distribution_entropy": dist_entropy,
                        "metric/output_token_length": np.mean(output_tolen),
                        "metric/thought_token_length": np.mean(thought_tolen),
                        "metric/valid_prob": np.mean(valid_prob),
                        "metric/episode_length": np.mean(episode_lengths),
                        "metric.self_bleu_score": torch.mean(rollouts.diversities).item(),
                        "loss/value.loss": value_loss,
                        "loss/action.loss": action_loss,
                        "loss/ppo.loss": ppo_loss,
                        "loss/sft.loss": sft_loss,
                        "action_log_prob": action_log_prob.to('cpu').float().numpy()[0],
                        "reward/format_reward.mean": rollouts.format_rewards.mean().item(),
                        "reward/reward.max": rollouts.rewards.max().item(),
                        "reward/reward.min": rollouts.rewards.min().item(),
                        "reward/reward.mean": rollouts.rewards.mean().item(),
                        "reward/reward.std": rollouts.rewards.std().item(),
                        "reward/reward.median": rollouts.rewards.median().item(),
                        "reward/return.max": rollouts.returns.max().item(),
                        "reward/return.min": rollouts.returns.min().item(),
                        "reward/return.mean": rollouts.returns.mean().item(),
                        "reward/return.std": rollouts.returns.std().item(),
                        "value/value.max": rollouts.value_preds.max().item(),
                        "value/value.min": rollouts.value_preds.min().item(),
                        "value/value.mean": rollouts.value_preds.mean().item(),
                        "value/value.std": rollouts.value_preds.std().item(),})

                # save checkpoint
                if accelerator.is_main_process:
                    print("Save model checkpoint...")
                    checkpoint_path = os.path.join(args.output_dir, args.tag, str(total_num_steps))     
                    os.makedirs(checkpoint_path, exist_ok=True)
                    actor_critic.value_model.base.save_pretrained(checkpoint_path)

    # save final model
    if accelerator.is_main_process: 
        print("Save final model...")
        checkpoint_path = os.path.join(args.output_dir, args.tag, "final")     
        os.makedirs(checkpoint_path, exist_ok=True)
        actor_critic.value_model.base.save_pretrained(checkpoint_path)

if __name__ == "__main__":
    main()
