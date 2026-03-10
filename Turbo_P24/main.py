from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()
print("using xformers")

import copy
import glob
import os
import time
from collections import deque

import gymnasium as gym
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage_sft import RolloutStorageSFT
from a2c_ppo_acktr.storage_kl import RolloutStorageKL

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from a2c_ppo_acktr.qwen_interface import find_target_linear_names, obtain_prompt_text
from a2c_ppo_acktr.qwen_interface import qwen_evaluate, qwen_generate

import math
import random
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
    ## environment interaction device is cpu
    model_device = device

    #initialization of model
    model_path = args.model_path
    cache_dir = args.cache_dir
    print(model_path)
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

    if "gym_cards" in args.env_name.lower():
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1)
    else:
        print("Environment not supported")
        exit(1)

    # hardcoding guider in GPU #1
    guider = MergedModelGuider(model_path, 1)

    obs = envs.reset()
    infos = envs.reset_infos
    ## Inputing Prompt here
    qs = get_prompt(args.env_name, args.action_only_prompt, infos)
    PROMPT_TEXT = obtain_prompt_text(image_processor, qs)

    projection_f = partial(text_projection, env_name=args.env_name)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             PROMPT_TEXT=PROMPT_TEXT,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_update_num=num_updates,
            max_grad_norm=args.max_grad_norm)

    if is_sft_guide:
        rollouts = RolloutStorageSFT(args.num_env_steps, args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.max_new_tokens)
    else:
        rollouts = RolloutStorageKL(args.num_env_steps, args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.max_new_tokens)

    _, _, output_ids, action, action_log_prob, action_tokens_log_prob, _, _ = actor_critic.act(obs, PROMPT_TEXT=PROMPT_TEXT)
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    if is_sft_guide:
        rollouts.obs_accu[0].copy_(obs)
    rollouts.to('cpu')

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)
    legal_action_prob = []
    episode_lengths = []
    valid_prob = []
    output_tolen = []
    thought_tolen = []
    answer_history = []

    start = time.time()
    if args.use_wandb:
        import wandb
        run_name = args.tag
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)

    # print(qs)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    num_explore = int(args.explore_portion*num_updates)
    prev_infos = []
    for j in tqdm(range(num_updates)):
        legal_action_prob.clear()
        valid_prob.clear()
        episode_lengths.clear()
        output_tolen.clear()
        thought_tolen.clear()
        episode_len = 0
        for step in range(args.num_steps):
            qs = get_prompt(args.env_name, args.action_only_prompt, infos)
            PROMPT_TEXT = obtain_prompt_text(image_processor, qs)
            print(f"-------- UPDATE {j+1}, STEP {step+1} --------")
            # print(f"PROMPT: {prompt}")
            # print(f"PREV_INFOS: {infos}")

            # Sample actions
            with torch.no_grad():
                value, INPUT_IDS, output_id, action, action_log_prob, action_tokens_log_prob, legal_action, raw_output_id = actor_critic.act(
                        rollouts.obs[step], PROMPT_TEXT = PROMPT_TEXT)
            legal_action_prob.append(legal_action)
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text_action = text_action.strip().removeprefix("```json").removesuffix("```").strip()
            answer_history.append(text_action)
            tolen = get_token_length(tokenizer, text_action)
            output_tolen.append(tolen)

            print(f"TEXT ACTION: {text_action}")
            print(f"INFOS: {infos}")
            img = rollouts.obs[step][0].cpu().numpy().astype(np.uint8)
            if is_sft_guide:
                correct_ids, labels, format_reward, valid, thought_token_length = guider.thought_guide_sft(PROMPT_TEXT, text_action, img)
                correct_ids = correct_ids.to(model_device)
                input_ids = torch.cat([INPUT_IDS, correct_ids], dim=1).to(model_device)
                padded_input_ids = torch.zeros(input_ids.size(0), 2*args.max_new_tokens).to(dtype=input_ids.dtype, device = input_ids.device)
                padded_input_ids[:, :input_ids.size(1)] = input_ids
                
                padded_labels = torch.full((1, 2*args.max_new_tokens), -100)
                padded_labels[:, INPUT_IDS.size(1): input_ids.size(1)] = labels
                padded_labels = padded_labels.to(input_ids.device)
            else:
                format_reward, valid, thought_token_length, kl_loss = guider.thought_guide_kl(PROMPT_TEXT, text_action, img, actor_critic.base, args)
            valid_prob.append(valid)
            if thought_token_length != None:
                thought_tolen.append(thought_token_length)
            
            prev_infos = copy.deepcopy(infos)
            obs, reward, done, infos = envs.step(action)
            episode_len += 1

            print(f"ACTION_REWARD: {reward.item()}")

            if abs(reward.item() + 1.0) < 1e-6:
                print("TRUNCATED!")
                done = [True]
                reward = torch.FloatTensor([-1.0])
                prev_infos = []
                obs = envs.reset()

            self_bleu_score = 0
            if done:
                infos = envs.reset_infos
                if len(answer_history) > 1:
                    self_bleu_score = get_self_bleu(answer_history)
                answer_history.clear()
                print("SELF_BLEU", self_bleu_score)
                episode_lengths.append(episode_len)
                episode_len = 0

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    if running_episode_rewards[i] > 0:
                        episode_success_rate.append(1)
                    else:
                        episode_success_rate.append(0)
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
            # bad_mask is a legacy implementation of the storage.py file
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            if is_sft_guide:
                rollouts.insert(obs, output_id, action,
                            action_log_prob, value, reward, masks, bad_masks, format_reward, padded_input_ids, padded_labels, self_bleu_score)
            else:
                rollouts.insert(obs, output_id, action,
                                action_log_prob, value, reward, masks, bad_masks, format_reward, self_bleu_score, kl_loss)
        print("****** iteration number:{} ******".format(j))
        # print("prompt:{}".format(prompt))
        print("text_action:{}".format(text_action))
        print("current observation:{}".format(prev_infos))
        print("ground truth:{}".format(infos))
        print("action log prob:{}".format(action_log_prob))
        print("action tokens log prob:{}".format(action_tokens_log_prob))
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1]).detach()

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
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}, legal_action_prob {:.2f}, dist_entropy {:.4f}, value_loss {:.4f}, action_loss {:.4f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate),
                        np.mean(legal_action_prob),
                        dist_entropy, value_loss, action_loss))
            
            if len(thought_tolen) == 0:
                thought_tolen.append(0)

            if args.use_wandb:
                wandb.log({"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "episode_reward.mean": np.mean(episode_rewards),
                        "episode_reward.median": np.median(episode_rewards),
                        "episode_reward.min": np.min(episode_rewards),
                        "episode_reward.max": np.max(episode_rewards),
                        "episode_success_rate.mean": np.mean(episode_success_rate),
                        "episode_action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                        "episode_length": np.mean(episode_lengths),
                        "prob.legal_action_prob": np.mean(legal_action_prob),
                        "prob.valid_prob": np.mean(valid_prob),
                        "metric.output_token_length": np.mean(output_tolen),
                        "metric.thought_token_length": np.mean(thought_tolen), 
                        "metric.self_bleu_score": torch.mean(rollouts.diversities).item(),
                        "distribution_entropy": dist_entropy,
                        "value.loss": value_loss,
                        "action.loss": action_loss,
                        "ppo.loss": ppo_loss,
                        "sft.loss": sft_loss,
                        "reward.format_reward": rollouts.format_rewards.mean().item(),
                        "reward.max": rollouts.rewards.max().item(),
                        "reward.min": rollouts.rewards.min().item(),
                        "reward.mean": rollouts.rewards.mean().item(),
                        "reward.std": rollouts.rewards.std().item(),
                        "reward.median": rollouts.rewards.median().item(),
                        "return.max": rollouts.returns.max().item(),
                        "return.min": rollouts.returns.min().item(),
                        "return.mean": rollouts.returns.mean().item(),
                        "return.std": rollouts.returns.std().item(),
                        "value.max": rollouts.value_preds.max().item(),
                        "value.min": rollouts.value_preds.min().item(),
                        "value.mean": rollouts.value_preds.mean().item(),
                        "value.std": rollouts.value_preds.std().item(),})

if __name__ == "__main__":
    main()

