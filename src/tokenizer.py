import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator


def read_texts_from_jsonl(file_path: str, max_lines=1000000) -> Generator[str, None, None]:
    """读取JSONL文件并安全提取文本数据"""
    # [NOTE] jsonl 文件的处理方式：打开文件，按行遍历，读取json，当对象用；
    # 为了安全，try-except, 处理异常，如jsonDecodeError, KeyError
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):  # [NOTE] 让计数直接从1开始，比用到i时都+1来的方便
            if i >= max_lines:
                break
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {i}") 
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding json in line {i}")
                continue
            except KeyError as e:
                print(e) 
                continue
        


def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    # [NOTE] 配置文件如何学习？能按模块划分来吗？但是并没有代码、函数那么好的逻辑，变成纯粹默写了
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        # 使用 Qwen2.5的 chat_template
        # [NOTE] 思考，为什么不用 LLaMA2 原装的？有什么问题吗？
        "chat_template": (  
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)



def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """训练并保存自定义tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.normalizer = NFKC()  # 添加文本规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()  # 为什么这里不用 BPFDecoder？

    # 配置特殊token
    special_tokens = [
        '<unk>',
        '<s>',
        '</s>', 
        '<|im_start|>',
        '<|im_end|>',
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # [NOTE] 教程为2，我们提高一些，防止垃圾词太多
        min_frequency=5,  # 提高低频词过滤频率，什么意思？
        show_progress=True,
        special_tokens=special_tokens,
        # [NOTE] 问题是，我怎么记得住要用哪个api呢？要先了解库吗，还是这部分可以让AI代劳？
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 训练tokenizer
    print(f"Train tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # 验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print(f"Special token mapping error: {e}")
        # [NOTE] 只有一个 raise 是什么意思？
        raise

    # 保存tokenizer文件
    # [NOTE] tokenizer 的实现是一个json文件吗？我只知道它的作用是切分文本，但是具体实现，以及为什么要训练？
    # 大致可以说是为了学习如何在预训练数据上分词。但是需要了解具体实现
    tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))
    
    # 创建配置文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")



def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    # [NOTE] IDE 对于 tokenizer 似乎没有自动提示。有哪些需要了解、记忆？比如 tokenizer 的基本信息吗？
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 测试特殊token处理
    # [NOTE] 为什么对特殊token的encode处理和上面的不一样？
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded) 
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)

    
if __name__ == '__main__':
    data_path = 'data/seq_monkey_datawhale.jsonl'
    save_dir = 'model/tokenizer'
    train_tokenizer(data_path, save_dir)

    eval_tokenizer(save_dir)

# [NOTE] 26.1.14 tokenizer train done!