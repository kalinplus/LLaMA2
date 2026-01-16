import os
# [NOTE] 应当看一眼数据集的大致情况，以及里面样本长什么样
# 下载预训练数据集
os.system("modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir /home/hkl/LLaMA2/data")
# 解压预训练数据集
os.system("tar -xvf /home/hkl/LLaMA2/data/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2")

# 下载SFT数据集
os.system(f'huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir /home/hkl/LLaMA2/data/BelleGroup')

# [NOTE] 26.1.13 download successfully