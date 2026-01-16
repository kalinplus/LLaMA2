import json
from tqdm import tqdm

def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    # 根据data中的item的不同来源(human/assistant)，为其分配不同的角色并添加到message里
    # 具体可以用 head -n 5 [path to sft data] 来查看
    messages = [
        {'role': 'system', 'content': '你是一个AI助手'},  # 后续的key都是role和content
    ]
    # sft data每行是一个json，这里应该是它的'conversation' key 对应的 value
    for item in data:
        if item['from'] == 'human':
            messages.append({'role': 'user', 'content': item['value']})
        if item['from'] == 'assistant':
            messages.append({'role': 'assistant', 'content': item['value']})
    return messages

sft_data_output_path = 'data/BelleGroup_sft.jsonl'
src_data_path = 'data/BelleGroup/train_3.5M_CN.json'

# 和 pretrain 差不多的方法
with open(sft_data_output_path, 'w', encoding='utf-8') as sft:
    with open(src_data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data, desc='Converting src sft data to std form...'):
            item = json.loads(line)
            messages = convert_message(item['conversations'])
            sft.write(json.dumps(messages, ensure_ascii=False) + '\n')

# [NOTE] 26.1.15 sft data preprocess done!