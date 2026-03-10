import torch
from copy import deepcopy

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[]):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    
    print(f"Found {len(lora_module_names)} lora modules.")
    return lora_module_names

def obtain_prompt_text(processor, qs):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": qs},
            ],
        },
    ]

    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def prepare_inputs_labels_for_multimodal(processor, image, prompt_text):
    image_inputs = processor.image_processor(images=[image], return_tensors='pt')
    text = deepcopy(prompt_text)
    merge_length = processor.image_processor.merge_size**2
    index = 0
    while processor.image_token in text:
        num_image_tokens = image_inputs["image_grid_thw"][index].prod() // merge_length
        text = text.replace(processor.image_token, "<|placeholder|>" * num_image_tokens, 1)
        index += 1
    text = text.replace("<|placeholder|>", processor.image_token)

    text_inputs = processor.tokenizer(text, return_tensors='pt')

    image_inputs = image_inputs.to('cuda')
    text_inputs = text_inputs.to('cuda')

    return text_inputs.input_ids, text_inputs.attention_mask, image_inputs.pixel_values, image_inputs.image_grid_thw