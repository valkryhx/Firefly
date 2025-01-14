import json
from loguru import logger
from torch.utils.data import Dataset
from glob import glob
import os
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.max_seq_length = max_seq_length

        if not (data_path is not None and os.path.exists(data_path)):
            raise ValueError("data_path requires a directory pointing to   jsons/jsonls")
        """https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py#L383"""
        data_files_list = glob(f'{data_path}/**/*.json', recursive=True) + glob(f'{data_path}/**/*.jsonl', recursive=True)
        logger.info(f"data files: {', '.join(data_files_list)}")

        #logger.info('Loading data: {}'.format(file))
        data_list = []
        for file_name in data_files_list :
            with open(file_name, 'r', encoding='utf8') as f:
                #logger.info(f"f.readlines() ={type(f.readlines())}")  # f.readlines()执行一次之后 再执行f.readlines() 就是空 一定注意
                data_list.extend( f.readlines() )
                
        logger.info("there are {} data in dataset".format(len(data_list)))
        #for item in data_list:
        #    print(item)
        #raise Error(123)
        self.data_list = data_list
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data) #注意文件中每行为json 这是对的
        conversation = data['conversation']

        # 收集多轮对话
        utterances = []
        for x in conversation:
            utterances.append(x['human'])
            utterances.append(x['assistant'])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [self.bos_token_id]
        target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs
