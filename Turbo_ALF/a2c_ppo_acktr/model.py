import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.qwen_interface import qwen_evaluate, qwen_generate
import torch.nn.init as init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VLMValue(nn.Module):
    """
    actually the base is also used for generation!
    """
    def __init__(self, base):
        super(VLMValue, self).__init__()
        self.base = base
        # hard-code to qwen hidden size for the value head
        self.value_head = nn.Sequential(
            nn.Linear(3584, 1024), # First layer
            nn.ReLU(), # Non-linearity
            nn.Linear(1024, 512), # Second layer
            nn.ReLU(), # Non-linearity
            nn.Linear(512, 1) # Output layer
            ).to(base.device, dtype=torch.float16) # Move to specified device with dtype

    def forward(self, processor, prompt_text, image_tensor):
        # image_tensor = image_tensor.to(self.base.device, dtype = self.base.dtype)
        inputs = processor(text=[prompt_text], images=[image_tensor], return_tensors='pt').to(self.base.device)
        
        outputs = self.base(
            input_ids = inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        values = self.value_head(hidden_states[-1][:, -1])
        return values


class VLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                image_processor,
                value_model,
                args,
                PROMPT_TEXT,
                projection_f,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(VLMPolicy, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.value_model = value_model
        self.base = value_model.base
        self.PROMPT_TEXT = PROMPT_TEXT
        self.projection_f = projection_f

    def act(self, inputs, deterministic=False, PROMPT_TEXT=None):
        if PROMPT_TEXT is None:
            PROMPT_TEXT = self.PROMPT_TEXT
        value, input_ids, output_ids, text_action, action_log_prob, action_tokens_log_prob, raw_output_ids = qwen_generate(
                                                    value_model = self.value_model,
                                                    processor = self.image_processor,
                                                    tokenizer = self.tokenizer,
                                                    prompt_text = PROMPT_TEXT,
                                                    image_tensor = inputs,
                                                    args = self.args)
        action = self.projection_f(text_action)
        return value, input_ids, output_ids, action, action_log_prob, action_tokens_log_prob, raw_output_ids

    def sft_forward(self, imgs, input_ids=None, labels=None):
        image_inputs = self.image_processor.image_processor(images=imgs, return_tensors='pt').to(self.base.device)
        attention_mask = torch.ones_like(input_ids).to(self.base.device)
        outputs = self.base(
            input_ids = input_ids,
            attention_mask=attention_mask,
            pixel_values=image_inputs.pixel_values,
            image_grid_thw=image_inputs.image_grid_thw,
            labels = labels,
            return_dict = True,
        )
        return outputs.loss, outputs.logits

    def get_value(self, inputs, PROMPT_TEXT=None):
        if PROMPT_TEXT is None:
            PROMPT_TEXT = self.PROMPT_TEXT
        return self.value_model(processor = self.image_processor, prompt_text = PROMPT_TEXT, image_tensor = inputs)

    def evaluate_actions(self, inputs, output_ids, PROMPT_TEXT=None):
        if PROMPT_TEXT is None:
            PROMPT_TEXT = self.PROMPT_TEXT
        value, action_log_prob, _ = qwen_evaluate(value_model = self.value_model,
                                        processor = self.image_processor,
                                        prompt_text = PROMPT_TEXT,
                                        output_ids = output_ids,
                                        image_tensor = inputs,
                                        temperature = self.args.temperature,
                                        thought_prob_coef = self.args.thought_prob_coef)
        return value, action_log_prob
