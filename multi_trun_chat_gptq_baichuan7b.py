from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    # model_name = 'YeungNLP/firefly-baichuan-7b-qlora-sft-merge'
    model_name = 'YeungNLP/firefly-ziya-13b-qlora-sft-merge'
    # model_name = 'YeungNLP/firefly-bloom-7b1-qlora-sft-merge'

    device = 'cuda'
    max_new_tokens = 500    # 每轮对话最多生成多少个token
    history_max_len = 1000  # 模型记忆的最大token长度
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # 记录所有历史记录
    history_token_ids = tokenizer('<s>', return_tensors="pt").input_ids

    # 开始对话
    user_input = input('User：')
    while True:
        user_input = '{}</s>'.format(user_input)
        user_input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        model_input_ids = history_token_ids[:, -history_max_len:].to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
            )
        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
        response = tokenizer.batch_decode(response_ids)
        print("Firefly：" + response[0].strip().replace('</s>', ""))
        user_input = input('User：')


if __name__ == '__main__':
    main()
