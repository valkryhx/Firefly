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
"""
单轮对话，不具有对话历史的记忆功能
"""


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
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda'
    input_pattern = '<s>{}</s>'
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
    while True:
        text = text.strip()
        text = input_pattern.format(text)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(text, "").replace('</s>', "").replace('<s>', "").strip()
        print("Bot：{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()
