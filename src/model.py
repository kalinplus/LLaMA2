import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from configs.config import ModelConfig
from .rope import apply_rotary_emb, precompute_freqs_cis



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    @param x: (bsz, seq_len, num_kv_heads, head_dim)
    @param n_rep: repetition times for kv to align with q
    @return y: (bsz, seq_len, num_q_heads=num_kv_heads * n_rep, head_dim)
    """
    # 将 k,v 的 num_kv_heads 维度通过 n_rep 扩展到和 q 相同的大小
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bsz, seq_len, num_kv_heads, head_dim = x.size() 
    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x 
    # 对张量进行扩展和重塑操作以重复键值对
        # 在第四个维度（头的维度前）添加一个新的维度
        # 将新添加的维度扩展到n_rep大小，实现重复的效果
        # 重新塑形，合并键/值对头的数量和重复次数的维度
    return (
        x.unsqueeze(3)  # (bsz, seq_len, num_kv_heads, 1, head_dim)
         .expand(-1, -1, -1, n_rep, -1)
         .reshape(bsz, seq_len, num_kv_heads * n_rep, head_dim)
    )



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps是为了防止除以0的情况
        self.eps = eps 
        # weight是一个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim)) 

    def _norm(self, x):
        # 计算RMSNorm的核心部分，即对x做norm处理
        # 首先计算输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        x_square_mean = torch.mean(x.pow(2), dim=-1, keepdim=True)  # [..., dim]->[..., 1]
        deno = torch.rsqrt(x_square_mean + self.eps)
        return x * deno  # [..., dim]
        
    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# RMSNorm test
args = ModelConfig()
norm = RMSNorm(args.dim, args.norm_eps)
x = torch.randn(1, 50, args.dim)
output = norm(x)

print(f"[INFO] Output shape: {output.shape}")
print("[INFO] Expected output shape: torch.Size([1, 50, 768])")
print("[INFO] RMSNorm test pass!")
# [NOTE] 26.1.1 pass



class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        n_kv_heads = args.n_kv_heads if hasattr(args, 'n_kv_heads') and args.n_kv_heads is not None else args.n_heads
        # 确保总头数可以被键值头数整除。
        assert args.n_heads % args.n_kv_heads == 0
        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.n_local_kv_heads = n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵。
        # [NOTE] 为什么上面算重复次数用local，这里又不用local？
        # [NOTE] 为什么bias=False？是为了教学目的（是什么），还是LLaMA2本就如此？
        self.wq = nn.Linear(args.dim, self.head_dim * args.n_heads, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * args.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * args.n_kv_heads, bias=False)
        # 输出权重矩阵。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout。
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率。
        self.dropout = args.dropout

        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）。
        # [NOTE] 稍微了解下 Flash Attention 的作用和实现
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))          
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            # [NOTE] 注册为缓冲区是什么意思？好像和参数化有点区别
            self.register_buffer('mask', mask)
    
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        """
        @param x: [batch_size, seq_len, dim]
        @param freqs_cos: [1, seq_len, 1, head_dim // 2], has reshaped for broadcast
        @param freqs_sin: [1, seq_len, 1, head_dim // 2], has reshaped for broadcast
        """
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape

        # 计算查询（Q）、键（K）、值（V）。
        # [batch_size, seq_len, self.head_dim * n_xxx_heads]
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 调整形状以适应头的维度。
        xq = xq.view(batch_size, seq_len, self.n_local_heads, -1)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, -1)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, -1)

        # 应用旋转位置嵌入（RoPE）。
        # [NOTE] 为什么是对xq, xk用旋转嵌入？
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 对键和值进行扩展以适应重复次数。
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理。
        # [NOTE] 为什么要这样做？应该与内存连续有关
        # [batch_size, self.n_local_heads, seq_len, self.head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持Flash Attention，选择实现方式。
        if self.flash:
            # 使用Flash Attention。
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout)
            # 使用手动实现的注意力机制。
        else:
            # -> [batch_size, n_heads, seq_len, seq_len]
            score = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim) 
            assert hasattr(self, 'mask')
            score += self.mask[:, :, :seq_len, :seq_len]
            attn = torch.softmax(score, dim=-1)
            attn = self.attn_dropout(attn)
            output = torch.matmul(attn, xv)  # [bsz, n_heads, seq_len, head_dim]

        # 恢复时间维度并合并头。
        output = output.transpose(2, 1).contiguous().view(batch_size, seq_len, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

# Test attn
# 创建Attention实例
attention_model = Attention(args)

# 模拟输入数据
batch_size = 1
seq_len = 50  # 假设实际使用的序列长度为50
dim = args.dim
x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
# freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
# freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

# 运行Attention模型
output = attention_model(x, freqs_cos, freqs_sin)

# attention出来之后的形状 依然是[batch_size, seq_len, dim]
print(f"[INFO] Output shape: {output.shape}")
print("[INFO] Expected output shape: torch.Size([1, 50, 768])")
print("[INFO] Attn test pass!")
# [NOTE] 26.1.8 Attn test pass



class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = hidden_dim * 2 // 3
            hidden_dim = hidden_dim // multiple_of * multiple_of  # 转换为最近的 multiple_of 的倍数
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) 
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) 
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) 
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        # [NOTE] 设计思路的问题。在 mlp 里引入逐元素乘法是想做什么？和传统的一路 linear+act 有什么区别？
        z1 = F.silu(self.w1(x))  # [bsz, slen, hidden_dim]
        z3 = self.w3(x)  # [bsz, slen, hidden_dim]
        return self.dropout(self.w2(z1 * z3))  # [bsz, slen, dim]


# 创建MLP实例
mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
# 随机生成数据
x = torch.randn(1, 50, args.dim)
# 运行MLP模型
output = mlp(x)
print(f"[INFO] mlp output shape: {output.shape}")
print("[INFO] Expected output shape: torch.Size([1, 50, 768])")
print("[INFO] mlp test pass!")
# [NOTE] 26.1.9 mlp test pass



class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads 
        # 定义输入维度
        self.dim = args.dim 
        # 定义每个头的维度，等于输入维度除以头数
        self.head_dim = args.dim // args.n_heads 
        # 定义LLaMA2Attention对象，用于进行多头注意力计算
        self.LLaMA2Attention = Attention(args)
        # 定义LLaMAMLP对象，用于进行前馈神经网络计算
        self.LLaMA2MLP = MLP(
            dim=args.dim, 
            hidden_dim=args.hidden_dim,
            multiple_of=args.n_heads,  # useless when hidden_dim passed
            dropout=args.dropout
        ) 
        # 定义层的ID
        self.layer_id = layer_id 
        # 定义注意力计算的归一化层
        self.attn_norm = nn.LayerNorm(normalized_shape=args.dim, eps=args.norm_eps) 
        # 定义前馈神经网络计算的归一化层
        self.mlp_norm = nn.LayerNorm(normalized_shape=args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.LLaMA2Attention(self.attn_norm(x), freqs_cos, freqs_sin)
        output = h + self.LLaMA2MLP(self.mlp_norm(x))
        return output


# 创建LLaMADecoderLayer实例
decoderlayer = DecoderLayer(0, args)

# 模拟输入数据
dim = args.dim
seq_len = 50

x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]

freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

out = decoderlayer(x, freqs_cos, freqs_sin)

print(f"[INFO] Decoder Layer out shape: {out.shape}") # 形状和输入的x一样 [batch_size, seq_len, dim]
print("[INFO] Expected out shape: torch.Size([1, 50, 768])")
print("[INFO] Decoder Layer test pass!")
# [NOTE] 一般准备测试的流程是什么？对于模型里的模块。准备好输入，观察输出是否符合期望？
# [NOTE] 26.1.9 Decoder Layer test pass



class Transformer(PreTrainedModel):
    config_class = ModelConfig  # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args 
        # 词汇表大小
        self.vocab_size = args.vocab_size 
        # 层数
        self.n_layers = args.n_layers 

        # 词嵌入层
        self.word_embedding = nn.Embedding(args.vocab_size, args.dim) 
        # Dropout层
        self.dropout = nn.Dropout(args.dropout) 
        # Decoder层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(i, args) for i in range(args.n_layers)
        ]) 
        # 归一化层
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps) 
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size) 

        # 将词嵌入层的权重与输出层的权重共享
        # [NOTE] 记录动机和原理
        self.output.weight = self.word_embedding.weight


        # 预计算相对位置嵌入的频率并注册为缓冲区
        # [NOTE] 回顾，缓冲区的作用
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim//args.n_heads, args.max_seq_len)
        self.register_buffer('freqs_cos', freqs_cos) 
        self.register_buffer('freqs_sin', freqs_sin)

        # 初始化所有权重
        # [NOTE] apply 的作用，为什么它一句就能初始化所有权重了？内部实现是类似遍历所有kv对吗？
        self.apply(self._init_weights) 

        # 对残差投影进行特殊的缩放初始化
        # 对应 GPT-2 提出的残差缩放初始化，用于防止残差累加导致输出方差爆炸
        # 2 * n_layers 是因为每层有 Attn 和 MLP 两个残差切入点
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*args.n_layers))

        # 初始化最后一次前向传播的损失属性
        # [NOTE] 这几个属性分别的作用是什么？
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表
        
    def _init_weights(self, module):
        # 初始化权重的函数
        # [NOTE] 其实整个 LLaMA2 中只有3种模块包含参数，nn.Linear, nn.Embedding 和 Norm 层
        # RMSNorm 层的参数在定义时初始化为 1 了，这里不用管
        # Attention, MLP 都是 nn.Linear 为基础组装起来的，只不过加上了 act, resid, dropout, RMSNorm 等等
        # [NOTE] normal_, zeros_ 的 _ 代表这些修改都是 inplace 的，与初始化的场景相适应
        if (isinstance(module, nn.Linear)):
            # [NOTE] 为什么初始化 std=0.02？看看文章了解下初始化相关的理论内容
            # 深度学习初始化理论（如 Xavier 初始化）通常认为，初始权重的方差应该与输入/输出维度的平方根成反比，以保持信号在传递过程中方差不变
            # 考虑维度为 768/1024，std 大概应当为 0.03，实际会再小一点，防止梯度爆炸。0.02 是一个经验值
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if (module.bias is not None):
                nn.init.zeros_(module.bias)
        elif (isinstance(module, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Optional[torch.Tensor], targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - kv_cache: bool, 是否使用键值缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        """
        # tokens, targets 可能存在于 kwargs，对应 key 为 input_ids 和 attention_mask，尝试获取
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        # print(f"[DEBUG] kwargs: {kwargs}")
        # print(f"[DEBUG] tokens shape: {tokens.shape}")
        # print(f"[DEBUG] targets shape: {targets.shape}")
        # 前向传播函数
        # 获取 bsz 和 seqlen
        _bsz, seqlen = tokens.shape
        
        # 通过词嵌入层和Dropout层
        h = self.word_embedding(tokens)
        # print(f"[DEBUG] word embeddings shape: {h.shape}")
        h = self.dropout(h)
        
        # 获取相对位置嵌入的频率
        # 要对 freqs_cos/sin 做一个适配 h 的 seqlen 的操作
        # [NOTE] 但是这样只能截断，是如何实现推理时长度的外推的呢？
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Decoder层
        for i, decoder in enumerate(self.decoder_layers):
            # print(f"[DEBUG] layer {i+1}")
            h = decoder(h, freqs_cos, freqs_sin) 
            
        # 通过归一化层
        h = self.norm(h) 

        # 如果给定了目标，计算损失
        # [NOTE] loss 怎么算？用什么损失函数？
        if targets is not None:
            logits = self.output(h)  # [bsz, seqlen, vocab_size]
            # [NOTE] targets 的形状是 [bsz, seqlen]？为什么训练时形状要这样变化？
            # [NOTE] 了解 cross_entropy 的用法
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none') 
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            # [NOTE] 为什么可以这样做？因为只需要预测下一个token？用 [-1] 切片是高级索引，可以保留这个维度
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('last_loss', self.last_loss)
        self.OUT.__setitem__('logits', logits)
        return self.OUT

    
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        # [NOTE] 代码多跑起来，多看维度变化，看各个操作的功能
        seqlen = idx.shape[1]
        for _ in range(max_new_tokens):
        # 如果序列上下文过长，截断它到最大长度
            if idx.shape[1] > self.args.max_seq_len:
                idx = idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx).logits
            logits = logits[:, -1, :]  # 只保留最后一个时间步的输出
            
            # 选择最有可能的索引
            # 这部分的目的就是得出 idx_next 变量
            if temperature == 0.0:
                # [NOTE] 了解 topk 返回 value, index 的应用场景。对比 gather 好像用于 logits 和 label 同时存在时？
                # [NOTE] 用于文本鉴伪场景如何呢？
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # [NOTE] temp!=0，不是贪心选择，此时怎么做？温度影响什么？值得了解一下
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    # 这时候应该要先选出topk概率的备选，然后怎么做？
                    # [NOTE] 答案：将更小的logits设置为 -inf
                    v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))  # [bsz, k]
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                # [NOTE] 有了概率，自然要采样，怎么做？值得了解下torch中的概率分布，以及大模型中常用的
                idx_next = torch.multinomial(probs, 1)
            

            # 将采样的索引添加到序列中并继续
            # [TODO] 这只在 bsz==1 时才会工作
            if idx_next == stop_id:
                break
            # [NOTE] 添加不能直接+，就是 cat/concat，凡是张量操作，都要考虑指定维度
            idx = torch.concat([idx, idx_next], dim=1)  # [bsz, seqlen+1]

        # 只返回生成的token
        # [NOTE] 对于自然语言描述，首先思考它翻译到代码是什么意思
        # 对于多种实现方式，仔细思考，甚至跑一跑，确认每一种行不行，有利于开阔思路
        # 很抽象，例子：只返回生成的token->从原始idx末尾开始切，只要后面的
        # 发散一下，这里用 idx[:, -max_new_tokens:] 行不行？并不行，因为上面 idx_next == stop_id 就停止生成了，不一定到最大
        return idx[:, seqlen:]  



# LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
x = torch.randint(0, 8192, (1, 50)) # [bs, seq_len]
# 实例化LLaMA2Model
model = Transformer(args=args)
# 计算model的全部参数
num_params = sum(p.numel() for p in model.parameters())
print('[INFO] Number of parameters:', num_params)

out = model(x)
print(f"[INFO] LLaMA2 logits shape: {out.logits.shape}") # [batch_size, 1, vocab_size]
print("[INFO] Expected logits shape: torch.Size([1, 1, 8192])")
print("[INFO] LLaMA2 forward test pass! (generate func have not tested)")

# [NOTE] 26.1.12 LLaMA2 forward (probably model) test pass