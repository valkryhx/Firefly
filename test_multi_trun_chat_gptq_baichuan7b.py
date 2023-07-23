from transformers import AutoTokenizer, BitsAndBytesConfig
#from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import (
    TaskType,
    LoraConfig,
    AdaLoraConfig ,  #  提出自2020年 感觉和lora区别不大 而且和qlora有冲突 这里代码没有用到 
                     #例子https://www.zhihu.com/question/596950521/answer/3109759716
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)
import argparse
from loguru import logger
import os
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict

from component.collator import SFTDataCollator
from component.dataset import SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss

from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
import json
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.utils.peft_utils import get_gptq_peft_model, GPTQLoraConfig
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.nn_modules.qlinear import GeneralQuantLinear
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_path", type=str, default='output/firefly-baichuan-7b/final/', help="")
    """这个type=bool 的值 只要写进python --use_safetensors A 无论A是什么都将让use_safetensors=True 这里也不能用type=str
       所以这里既然已经默认use_safetensors=False 那在传参时就不要传入这个参数了！很奇怪 
    """
    #parser.add_argument("--use_safetensors", type=bool, default=False, help="If the original GPTQ model is saved in .safetensors format ,then set this to True")
    parser.add_argument("--use_safetensors",type=eval, 
                      choices=[True, False], 
                      default='True')   #不能用type=bool  否则不能用 --xxx Treu传值。type=bool 只能  --xx传值  提示性不想 参考 https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument("--base_model_name_or_path" ,type=str, default="fireballoon/baichuan-vicuna-chinese-7b-gptq")
    args = parser.parse_args()
    # model_name = 'YeungNLP/firefly-baichuan-7b-qlora-sft-merge'
    #model_name = "TheBloke/baichuan-7B-GPTQ"     #'TheBloke/baichuan-7B-GPTQ'
    model_name = args.base_model_name_or_path #"fireballoon/baichuan-vicuna-chinese-7b-gptq"
    logger.info(f"args.use_safetensors= {args.use_safetensors} , {type(args.use_safetensors)}")
    device = 'cuda'
    max_new_tokens = 500    # 每轮对话最多生成多少个token
    history_max_len = 4000  # 模型记忆的最大token长度
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        device_map='auto',
        low_cpu_mem_usage = True ,
        #max_memory= max_memory ,  # MB ,max_memory,
        trust_remote_code=True,
        inject_fused_attention = True,
        inject_fused_mlp = False,
        use_triton=True,
        warmup_triton=False,
        trainable=False,
        use_safetensors = args.use_safetensors,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    model = get_gptq_peft_model(model=model,model_id=args.peft_path,train_mode=False)
    print(model)
    text = input('User：')
    # 记录所有历史记录
    history_token_ids = tokenizer('<s>', return_tensors="pt").input_ids

    # 开始对话
    user_input = input('User：')
    while True:
        print("进入循环")
        user_input = user_input.strip()
        user_input = '{}</s>'.format(user_input)
        user_input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        print("step 0")
        model_input_ids = history_token_ids[:, -history_max_len:].to(device)
        print("step 1")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
            )
        print("step 2")
        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
        response = tokenizer.batch_decode(response_ids)
        print("step 3")
        print("Bot：" + response[0].strip().replace('</s>', ""))
        user_input = input('User：')


if __name__ == '__main__':
    main()
