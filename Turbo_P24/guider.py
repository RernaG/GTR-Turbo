import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, LogitsProcessorList
from peft import PeftModel, get_peft_model_state_dict
import os
from metric import get_token_length
import copy
import json
import traceback
from collections import defaultdict
import random
from json_repair import repair_json

class SingleLineArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        result = super().encode(obj)
        result = result.replace('\"\\\"', '').replace('\\\"\"', '')
        return result


def extract_json_object(text):
    raw = text.strip()
    for fence in ("```json", "```JSON", "```"):
        raw = raw.replace(fence, "")
    raw = raw.strip()

    first_object = raw.find("{")
    first_array = raw.find("[")
    start_candidates = [idx for idx in (first_object, first_array) if idx != -1]
    if not start_candidates:
        raise json.JSONDecodeError("No JSON object found", raw, 0)

    start = min(start_candidates)
    candidate = raw[start:]
    decoder = json.JSONDecoder(object_pairs_hook=dict)
    try:
        data, _ = decoder.raw_decode(candidate)
    except json.JSONDecodeError:
        repaired = repair_json(candidate)
        data = json.loads(repaired, object_pairs_hook=dict)

    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict) and "current formula" in data and "formula" not in data:
        data["formula"] = data.pop("current formula")
    return data

def preprocess_data(data):
    assert isinstance(data, dict)
    if "cards" in data.keys():
        data["cards"] = '"' + str(data["cards"]) + '"'
    return data

class MergedModelGuider():
    def __init__(self, model_path, gpu_id):
        self.base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
        ).to(device=f'cuda:{gpu_id}', dtype=torch.bfloat16)
        self.model = self.base
        self.adapter_names = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.image_processor = AutoProcessor.from_pretrained(
            model_path,
        )
        self.merged_cnt = 0

    def merge(self, ckpt_path, tag, update_num, use_ema=False, ema_alpha=0.5):
        adapter_names = []
        for j in range(4):
            path = os.path.join(ckpt_path, tag, "update_" + str(update_num) + "_" + str(j))
            print("Loading adapter: update_" + str(update_num) + "_" + str(j))
            if self.merged_cnt == 0:
                self.model = PeftModel.from_pretrained(self.base, path, adapter_name="update_" + str(update_num) + "_" + str(j), low_cpu_mem_usage=True, torch_device='cpu')
            else:
                _ = self.model.load_adapter(path, adapter_name="update_" + str(update_num) + "_" + str(j), low_cpu_mem_usage=True, torch_device='cpu')
            adapter_names.append("update_" + str(update_num) + "_" + str(j))
            self.merged_cnt += 1
        
        print("Merging adapters...")
        if update_num == 0:
            weights = [0.25, 0.25, 0.25, 0.25]
        else:
            if use_ema:
                weights = [ema_alpha/4, ema_alpha/4, ema_alpha/4, ema_alpha/4, 1-ema_alpha]
            else:
                weights = [1/(self.merged_cnt+4), 1/(self.merged_cnt+4), 1/(self.merged_cnt+4), 1/(self.merged_cnt+4), self.merged_cnt/(self.merged_cnt+4)]
        adapter_name = "merge"
        self.model.add_weighted_adapter(adapter_names, weights, adapter_name, combination_type="ties", density=0.8)
        self.model.set_adapter("merge")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name in adapter_names:
            if name != "merge":
                self.model.delete_adapter(name)
    
    def generate_thought(self, obs, prompt_text, temperature=0.3):
        inputs = self.image_processor(text=[prompt_text], images=[obs], return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=256,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                top_k=5,
            )
            output_ids = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_ids = torch.stack(output_ids)
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        final_answer = repair_json(answer)
        print("GUIDER_ANSWER:", answer)
        return final_answer
    
    def selective_log_softmax(self, logits, index):
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps
    
    def dual_forward(self, agent_model, input_ids, sequence, pad_token_id):
        with torch.inference_mode():
            inputs = torch.cat((input_ids, sequence), axis=1)
            attention_mask = inputs != pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long()
            inputs = torch.masked_fill(inputs, ~attention_mask, 0)
            
            agent_outputs = agent_model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )

            inputs = inputs.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            position_ids = position_ids.to(self.model.device)

            guider_outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )
        return agent_outputs, guider_outputs

    def get_kl_loss(self, agent_model, obs, prompt_text, agent_sequence, args):
        inputs = self.image_processor(text=[prompt_text], images=[obs], return_tensors='pt').to(agent_model.device)
        context_length = inputs.input_ids.shape[1]
        agent_sequence = torch.tensor(agent_sequence).unsqueeze(0).to(agent_model.device)
        agent_outputs, guider_outputs = self.dual_forward(agent_model, inputs.input_ids, agent_sequence, self.tokenizer.pad_token_id)
        agent_logits = agent_outputs.logits[:, context_length:]
        guider_logits = guider_outputs.logits[:, context_length:]
        
        # logprob
        agent_log_probs = self.selective_log_softmax(agent_logits, agent_sequence)
        agent_sequence = agent_sequence.to(guider_logits.device)
        agent_log_probs = agent_log_probs.to(guider_logits.device)
        guider_log_probs = self.selective_log_softmax(guider_logits, agent_sequence)

        # compute kl
        kl = agent_log_probs.float() - guider_log_probs.float()
        kl = kl.clamp(min=0, max=10)
        return kl.mean().item()

    def extract_thought(self, text_action):
        check_keys = ['cards', 'formula', 'thoughts', 'action']
        try:
            data = extract_json_object(text_action)
            if set(check_keys).issubset(set(data.keys())):
                action = data['action']
                del data['action']
                return json.dumps(data), True, action
            return "", False, None
        except Exception as e:
            traceback.print_exc()
            return "", False, None

    def replace_action(self, text, target_action):
        try:
            data = extract_json_object(text)
            if 'action' in data.keys():
                action = data['action']
                data['action'] = target_action
                return data, True, action
            return "", False, None
        except Exception as e:
            traceback.print_exc()
            return "", False, None

    def thought_guide_sft(self, prompt_text, text_action, img):
        thought, valid, raw_action = self.extract_thought(text_action)
        print(f"THOUGHT: {thought}, {valid}")
        format_reward = 3

        if not valid:
            labels = torch.tensor([[-100]])
            correct_tok = []
            format_reward = -2
            thought_token_length = None
        else:
            thought_token_length = get_token_length(self.tokenizer, thought)

            ok = False
            tmp = 0.2
            max_retries = 20
            for _retry in range(max_retries):
                try:
                    guide_thought = self.generate_thought(img, prompt_text, tmp)
                    correction, ok, action = self.replace_action(guide_thought, raw_action)
                    if ok:
                        break
                except Exception:
                    pass
                tmp = min(tmp * 1.1, 0.9)
            if not ok:
                print(f"WARNING: guider failed after {max_retries} retries")
                return torch.tensor([[]]).long(), torch.tensor([[-100]]), -2, False, None

            # post processing
            processed_data = preprocess_data(correction)
            correct_res = json.dumps(processed_data, cls=SingleLineArrayEncoder, indent=2)
            print("processed_guidance", correct_res)

            action_res = f'"action": "{raw_action}"'
            correct_tok = self.tokenizer.encode(correct_res)
            labels = copy.deepcopy(correct_tok)
            action_tok = self.tokenizer.encode(action_res)
            len_a_t = len(action_tok)
            for i in range(len(labels) - len_a_t, 0, -1):
                if labels[i:i + len_a_t] == action_tok:
                    labels[i:i + len_a_t] = [-100] * len_a_t
                    break
            labels = torch.tensor(labels).unsqueeze(0)
        
        return torch.tensor(correct_tok).unsqueeze(0), labels, format_reward, valid, thought_token_length

    def thought_guide_kl(self, prompt_text, text_action, img, agent_model, args):
        thought, valid, raw_action = self.extract_thought(text_action)
        print(f"THOUGHT: {thought}, {valid}")
        format_reward = 3

        if not valid:
            labels = torch.tensor([[-100]])
            format_reward = -2
            thought_token_length = None
            kl_loss = 100
        else:
            thought_sequence = self.tokenizer.encode(thought)
            thought_token_length = len(thought_sequence)
            kl_loss = self.get_kl_loss(agent_model, img, prompt_text, thought_sequence, args)

        print("KL_LOSS", kl_loss)
        
        return format_reward, valid, thought_token_length, kl_loss
