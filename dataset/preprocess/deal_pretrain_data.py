from tqdm import tqdm
import json

# 1. 处理预训练数据
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

input_file = 'data/mobvoi_seq_monkey_general_open_corpus.jsonl'

# 读取jsonl，然后逐行使用 split_text 切分
with open('seq_monkey_datawhale.jsonl', 'w', encoding='utf-8') as pretrain:
    with open(input_file, 'r', encoding='utf-8') as f:
        # [NOTE] 这里书中为了方便，直接全部读取。10GB左右的数据，可能占用多少内存/显存？
        # 服务器上没关系。如果是个人笔记本，通常不够，有没有更好的读取方法？
        data = f.readlines() 
        # [NOTE] jsonl 文件，每行是一个json。适用于文件很大的情况，比较小直接用json
        for line in tqdm(data, desc='Preprocess pretrain dataset'):
            # [NOTE] json.load 和 json.loads 有什么区别？其中的常用参数？
            line = json.loads(line)
            text = line['text']  # 这要看看具体数据格式其实才知道怎么做
            chunks = split_text(text)
            for chunk in chunks:
                # [NOTE] 同样的，json.dump 和 json.dumps？其中的常用参数？
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# [NOTE] 26.1.13 pretrain dataset preprocess done!