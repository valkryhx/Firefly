# 参考 https://huggingface.co/TheBloke/guanaco-33B-GPTQ/discussions/10

# run with
# python simple_autogptq.py ./text-generation-webui/models/TheBloke_guanaco-33B-GPTQ/ 
#   --model_basename Guanaco-33B-GPTQ-4bit.act-order 
#   --use_safetensors 
#   --use_triton

import os
import argparse
from peft import prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, get_gptq_peft_model, BaseQuantizeConfig
from auto_gptq.utils.peft_utils import GPTQLoraConfig

parser = argparse.ArgumentParser(description='Simple AutoGPTQ example')
parser.add_argument('model_name_or_path', type=str, help='Model folder or repo')
parser.add_argument('--model_basename', type=str, help='Model file basename if model is not named gptq_model-Xb-Ygr')
parser.add_argument('--use_slow', action="store_true", help='Use slow tokenizer')
parser.add_argument('--use_safetensors', action="store_true", help='Model file basename if model is not named gptq_model-Xb-Ygr')
parser.add_argument('--use_triton', action="store_true", help='Use Triton for inference?')
parser.add_argument('--bits', type=int, default=4, help='Specify GPTQ bits. Only needed if no quantize_config.json is provided')
parser.add_argument('--group_size', type=int, default=128, help='Specify GPTQ group_size. Only needed if no quantize_config.json is provided')
parser.add_argument('--desc_act', action="store_true", help='Specify GPTQ desc_act. Only needed if no quantize_config.json is provided')
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name_or_path = args.model_name_or_path
model_basename = args.model_basename
tokenizer_name_or_path = model_name_or_path

peft_config = GPTQLoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          use_fast=not args.use_slow,
                                          unk_token="<unk>",
                                          bos_token="<s>",
                                          eos_token="</s>")
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path)
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    #model_basename=model_basename,
    #use_safetensors=args.use_safetensors,
    use_triton=args.use_triton,
    device="cuda:0",
    trainable=True,
    inject_fused_attention=True,
    inject_fused_mlp=False,
    #quantize_config=quantize_config
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_gptq_peft_model(model, peft_config=peft_config, auto_find_all_linears=True, train_mode=True)
model.print_trainable_parameters()

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
data = data['train'].train_test_split(train_size=0.9, test_size=0.1)

tokenizer.pad_token = tokenizer.eos_token
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data['test'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=3,
        learning_rate=2e-2,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        evaluation_strategy='steps',
        eval_steps=1,
        report_to="tensorboard"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()
trainer.model.save_pretrained(hf_train_args.output_dir)
