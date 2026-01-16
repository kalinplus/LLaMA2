from torch.utils.data import Dataset
import json
import numpy as np
import torch

class PretrainDataset(Dataset):
    """
    Pretrain Dataset 主要是将 text 通过 tokenizer 转换成 input_id，然后将 input_id 拆分成 X 和 Y，
    其中 X 为 input_id 的前 n-1 个元素，Y 为 input_id 的后 n-1 个元素
    loss_mask 主要是用来标记哪些位置需要计算损失，哪些位置不需要计算损失

    例如：max_length = 9
    输入序列：[BOS, T1, T2, T3, T4, T5, T6, T7, EOS]
    样本拆分：
        X：[BOS, T1, T2, T3, T4, T5, T6, T7] → 模型输入上下文
        Y：[T1, T2, T3, T4, T5, T6, T7, EOS] → 模型预测目标
    损失掩码：
    有效位置：[0, 1, 1, 1, 1, 1, 1, 1, 1] → 仅对T1-EOS计算损失
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0  # [NOTE] padding 的作用？
        # 预计算每行的起始字节偏移量
        # [NOTE] 这部分在统计行数？为什么这样可以统计行数？统计出来做什么？
        self._offsets = []
        # [NOTE] 了解python中基本的文件操作和api，更重要的是，什么时候用什么？
        # 这里用b模式做什么？tell呢？
        with open(data_path, 'rb') as f:
            self._offsets.append(0)  # 这只是占掉index=0
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1  # 最后一个 tell() 是 EOF

    def __len__(self):
        return self._total_lines 

    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            # [NOTE] 同样的，对这种基于stream的操作，很陌生
            f.seek(self._offsets[index])
            line = f.readline().decode(encoding='utf-8')
        sample = json.loads(line)
        # 这部分是在构造text及其对应的input_ids
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        # 这部分主要是api，tokenizer约定的输出格式，有个印象先
        input_ids = self.tokenizer(text).data['input_ids'][:self.max_length]  # 这里怕文本太长
        text_len = len(input_ids)
        # 没满最大长度的剩余部分。这里又担心文本不够长
        padding_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        # [NOTE] 这里的转换到64是何意味？正常来说转到32不就够了吗？ROPE那里也是
        # 以及结合训练时的情况，是如何用数据实现并行训练的？目标是流畅的讲出全过程
        # 转换到np是为了转成torch.tensor？之前是列表，不能直接转？应该也可以
        input_ids = np.array(input_ids)
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)