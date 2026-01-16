import math

def get_lr(it, all, args):
    """
    计算当前迭代的学习率，使用余弦退火调度策略
    
    学习率调度策略：
    1. Warmup阶段：学习率从0线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率
    
    Args:
        it (int): 当前迭代步数
        all (int): 总迭代步数
        args: 模型参数
        
    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = args.learning_rate / 10  # 最小学习率，为初始学习率的1/10

    # Warmup阶段：线性增长
    if it < warmup_iters:
        return it / warmup_iters * args.learning_rate
    
    # 超出训练步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    # 余弦退火阶段
    decay_radio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # 映射当前步数到[0, 1]，当前到了余弦调度的百分之几了
    cos_coeff = math.cos(decay_radio * math.pi)  # 余弦系数，范围为 [-1, 1]
    cos_coeff = 0.5 * (1 + cos_coeff)  # 线性映射到 [0, 1]
    lr = min_lr + cos_coeff * (args.learning_rate - min_lr)  # 线性插值映射
    return lr