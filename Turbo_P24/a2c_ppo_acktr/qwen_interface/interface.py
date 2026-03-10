import torch

def qwen_generate(value_model, processor, tokenizer, prompt_text, image_tensor, args):
    base = value_model.base
    inputs = processor(text=[prompt_text], images=[image_tensor], return_tensors='pt').to(base.device)
    with torch.inference_mode():
        generated_ids = base.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            top_k=5,
        )
        output_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_ids = torch.stack(output_ids)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    padded_output_ids = torch.zeros(output_ids.size(0), 2*args.max_new_tokens).to(dtype=output_ids.dtype, device = output_ids.device)
    padded_output_ids[:, :output_ids.size(1)] = output_ids
    with torch.no_grad():
        values, sum_log_probs, action_tokens_log_prob = qwen_evaluate(value_model, processor, prompt_text, padded_output_ids, image_tensor, args.temperature, args.thought_prob_coef)
    return values, inputs.input_ids, padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob, output_ids

def qwen_evaluate(value_model, processor, prompt_text, output_ids, image_tensor, temperature, thought_prob_coef):
    base = value_model.base
    inputs = processor(text=[prompt_text], images=[image_tensor], return_tensors='pt').to(base.device)
    input_ids = inputs.input_ids
    if output_ids.size(0) != 1:
        input_ids = input_ids.broadcast_to(output_ids.size(0), input_ids.size(-1))
    input_ids = torch.cat([input_ids, output_ids], dim = 1)
    attention_mask = torch.ones_like(input_ids)
    outputs = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs.pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        output_hidden_states = True,
    )
    scores = outputs.logits

    input_token_len = input_ids.shape[1] - output_ids.shape[1]
    hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
    values = value_model.value_head(hidden_states)
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    output_ids_mask = (output_ids != 0)[:, 1:]
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    unfolded = output_ids.unfold(dimension=-1, size=2, step=1)
    _action_token_ids = processor.tokenizer.encode('"action":', add_special_tokens=False)
    target = torch.tensor(_action_token_ids[-2:]).to(base.device)
    matches = (unfolded == target).all(dim = -1)
    match_index = matches.nonzero(as_tuple=True)[-1]
    if match_index.shape[0] >= 1:
        match_index = match_index[-1].unsqueeze(0)
    else:
        try:
            match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        except:
            sum_log_prob = torch.tensor([-2]).to(base.device)
            action_tokens_log_prob = torch.tensor([-1]).to(base.device)
            return values, sum_log_prob, action_tokens_log_prob
    ## omitting the second token for calculating log prob, because its logprb is very very small
    thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
    action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    return values, sum_log_prob, action_tokens_log_prob
