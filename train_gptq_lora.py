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


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    #cls = bnb.nn.Linear4bit
    cls = GeneralQuantLinear  #if not(args.full_finetune) else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if  'output_layer' in lora_module_names:  # # needed for 16-bit ,chatglm2-6b use output_layer instead of lm_head
        lora_module_names.remove('output_layer')
    return list(lora_module_names)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/qlora/baichuan-sft-qlora.json', help="")
    parser.add_argument("--deepspeed", type=str, default="train_args/qlora/ds_zero2_config.json")
    parser.add_argument("--peft_path", type=str, default="output/lora_baichuan")
    parser.add_argument("--output_dir", type=str, default="output dir")
    parser.add_argument("--ddp_find_unused_parameters", type=str, default=False)
    #parser.add_argument('--ddp_find_unused_parameters', default=False, action=argparse.BooleanOptionalAction)  # 这种写法虽然更正常 但是好像不能满足 --ddp True/False的明确输入要求
    # local_rank 要加入argument ，因为使用deepspeed会传入这个参数 不加的话会报错 unrecognized argument
    # 参考我写的chatGLM-6B-QLoRA/train_qlora_deepspeed_zero.py
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argumente("--use_safetensors",type=str,default=True)   #不能用type=bool  否则不能用 --xxx Treu传值。type=bool 只能  --xx传值  提示性不想 参考 https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse 
    parser.add_argument("--model_name_or_path",type=str,default="fireballoon/baichuan-vicuna-chinese-7b-gptq")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    peft_path = args.peft_path
    use_safetensors = args.use_safetensors  # 
    model_name_or_path = args.model_name_or_path
    output_dir_overwrite = args.output_dir  #  命令行优先级高 下面会覆盖从json中读取的参数
    """find_unused_parameters 极其重要 必须为False才能DDP分布式训练"""
    ddp_find_unused_parameters = args.ddp_find_unused_parameters
    logger.info(f"args.ddp_find_unused_parameters={args.ddp_find_unused_parameters}")
    logger.info(f"mark 0   ddp_find_unused_parameters = {ddp_find_unused_parameters}")
    #here we get deepspeed config  before the args changed by the line#92 args, training_args = parser.parse_json_file(json_file=train_args_file)
    with open(args.deepspeed,'r',encoding='utf-8') as fr:   # 这里就是向TrainingArgs中添加deepseed字段
        training_args_deepspeed = json.load(fr)  # set trainingArgs中deepspeed=ds_config
        
    # 读取训练的参数配置
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，并且用上面由命令行传入的参数新增或者替换json文件中的自带参数 因为命令行参数优先级高
    lora_args, training_args = parser.parse_json_file(json_file=train_args_file)
    training_args.deepspeed = training_args_deepspeed
    training_args.peft_path = peft_path
    training_args.use_safetensors = use_safetensors
    training_args.model_name_or_path =model_name_or_path
    logger.info(f"mark 1   training_args.model_name_or_path = {training_args.model_name_or_path}")
    logger.info(f"mark 1   training_args.use_safetensors = {training_args.use_safetensors}")
    logger.info(f"mark 1   training_args.ddp_find_unused_parameters = {training_args.ddp_find_unused_parameters}")
    training_args.ddp_find_unused_parameters  = ddp_find_unused_parameters
    logger.info(f"mark 2   training_args.ddp_find_unused_parameters = {training_args.ddp_find_unused_parameters}")
    # 创建输出目录
    training_args.output_dir = output_dir_overwrite #  命令行优先级高 覆盖从json中读取的参数
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir,exist_ok=True)
    # logger.add(join(training_args.output_dir, 'train.log'))
    # logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    logger.info(f"mark 3   training_args.ddp_find_unused_parameters = {training_args.ddp_find_unused_parameters}")
    return lora_args, training_args


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')

    """
    # 下面的设置至关重要，否则无法多卡训练
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    training_args.ddp_find_unused_parameters = False
    device_map = "auto"
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
   """
    ds_config ={
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": False
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
            "adam_w_mode": True
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        },
        "offload_param": {
            "device": "none",
            "pin_memory": False
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e6,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e6,
        "stage3_max_reuse_distance": 1e6,
        "stage3_gather_16bit_weights_on_model_save": True
    },    
    "train_batch_size": 8 ,  
    "train_micro_batch_size_per_gpu":4
}
    #dschf = HfDeepSpeedConfig(ds_config)
    # 加载模型
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     #device_map=device_map,
    #     #device_map="auto",  # DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`.
    #     load_in_4bit=True,
    #     torch_dtype=torch.float16,
    #     trust_remote_code=True,
    #     empty_init=False, 
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         #llm_int8_threshold=6.0,
    #         #llm_int8_has_fp16_weight=False,
    #     ),
    # )

    n_gpus = torch.cuda.device_count()
    max_memory = f'{12000}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    logger.info(f"GPTQ 模型加载开始: {args.model_name_or_path}")
    model = AutoGPTQForCausalLM.from_quantized(
        args.model_name_or_path,
        device_map='auto',
        low_cpu_mem_usage = True ,
        #max_memory= max_memory ,  # MB ,max_memory,
        trust_remote_code=True,
        inject_fused_attention = True,
        inject_fused_mlp = False,
        use_triton=True,
        warmup_triton=False,
        trainable=True,
        use_safetensors = training_args.use_safetensors #False #True
    )
    model.model.quantize_config = model.quantize_config
    model.train()
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    # fp32 is really important during training!
    model.config.torch_dtype=torch.float16   #(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model.config.use_cache = False
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # 部分tokenizer没有pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    # 如果两者相同，模型训练时不会计算eos_token_id的loss
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise Exception('pad_token_id should not be equal to eos_token_id')

    # casts all the non int8 modules to full precision (fp32) for stability
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    logger.info(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    # 找到所有需要插入adapter的全连接层
    target_modules = find_all_linear_names(model)
    # 初始化lora配置
    # config = LoraConfig(
    #     r=args.lora_rank,
    #     lora_alpha=args.lora_alpha,
    #     target_modules=target_modules,
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    logger.info("GPTQ 模型加载完成")
    config = GPTQLoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
       target_modules=["W_pack", "o_proj"],#target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    #model = get_peft_model(model, config)
    model = get_gptq_peft_model(model, config, auto_find_all_linears=False, train_mode=True)

    # 
    peft_path = training_args.peft_path
    if peft_path is not None:
        checkpoint_name = os.path.join(peft_path, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                peft_path, 'adapter_model.bin'
            )
            resume_from_checkpoint = False   #注意这个resume_from_checkpoint 本来是trainer.train（）的参数 用于指示是否继续上一次的训练 取值是True/False 本来不是给peft用的 这里原始的代码是为了兼容 写的有点儿乱
        if os.path.exists(checkpoint_name):
            logger.info(f'Continue training  from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f'Checkpoint {checkpoint_name} not found')
    
    model.print_trainable_parameters()
    #model.config.torch_dtype = torch.float16 #torch.float32

    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if args.bf16:
    #             module = module.to(torch.bfloat16)
    #     if 'norm' in name:
    #         module = module.to(torch.float32)
    #     if 'lm_head' in name or 'embed_tokens' in name:
    #         if hasattr(module, 'weight'):
    #             if args.bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    logger.info("GPTQ LORA模型加载完成")
    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    # 初始化损失函数
    loss_func = TargetLMLoss(ignore_index=tokenizer.pad_token_id)

    # 加载训练集
    train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    logger.info(training_args)
    training_args.ddp_find_unused_parameters  = False
    logger.info(f"after change training_args.ddp_find_unused_parameters  = {training_args.ddp_find_unused_parameters }")
    # 初始化Trainer
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func
    )
    return trainer


def main():
    # 进行一些配置和检查
    myargs, training_args = setup_everything()
    logger.info(f"myargs={myargs}")
    # 加载各种组件
    logger.info(f"original training_args.ddp_find_unused_parameters  = {training_args.ddp_find_unused_parameters }")
    trainer = init_components(myargs, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
