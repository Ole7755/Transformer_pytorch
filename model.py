# model.py
#
# 该文件定义了Transformer模型的完整结构。
# 遵循"Attention Is All You Need"论文，并结合了常见的实践（如Pre-LN），它由以下几个核心组件构成：
# - InputEmbeddings: 将输入的token ID转换为词向量。
# - PositionalEncoding: 为模型注入序列的位置信息。
# - LayerNormalization: 层归一化，用于稳定训练。
# - FeedForwardBlock: 前馈神经网络，是Transformer块的一部分。
# - MultiHeadAttentionBlock: 多头注意力机制，模型的核心。
# - ResidualConnection: 残差连接，帮助梯度传播（采用Pre-LN结构）。
# - EncoderBlock & DecoderBlock: 编码器和解码器的基本构建单元。
# - Encoder & Decoder: 由多个Block堆叠而成。
# - ProjectionLayer: 将解码器的输出映射到词汇表大小，用于生成预测。
# - Transformer: 整合编码器、解码器和投影层，形成完整的模型。

import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    输入嵌入层

    该模块负责将输入的离散词元 (token IDs) 转换为连续的密集向量表示 (embeddings)。
    这是模型处理文本数据的第一步。
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        初始化嵌入层。

        Args:
            d_model (int): 模型的维度，也是嵌入向量的维度。
            vocab_size (int): 词汇表的大小，即不同词元的总数。
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # nn.Embedding 是一个高效的查找表，存储了 vocab_size 个大小为 d_model 的嵌入向量。
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程。

        Args:
            x (torch.Tensor): 输入的词元ID张量, 形状为 (batch_size, seq_len)。

        Returns:
            torch.Tensor: 转换后的嵌入向量张量，形状为 (batch_size, seq_len, d_model)。
        """
        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """
    位置编码层

    Transformer本身不包含任何关于序列顺序的信息。该模块通过注入位置信号来解决这个问题。
    它使用不同频率的正弦和余弦函数来创建位置编码，并将其加到输入嵌入上。
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        初始化位置编码层。

        Args:
            d_model (int): 模型的维度。
            seq_len (int): 序列的最大长度。
            dropout (float): Dropout的比率。
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # 创建一个大小为 (seq_len, d_model) 的零矩阵，用于存放位置编码
        pe = torch.zeros(seq_len, d_model)

        # 创建一个表示位置的张量 (0, 1, ..., seq_len-1)
        # 形状: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # 计算除数项，用于缩放不同维度的频率
        # div_term 的形状: (d_model / 2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 使用正弦函数填充偶数索引的维度
        pe[:, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数填充奇数索引的维度
        pe[:, 1::2] = torch.cos(position * div_term)

        # 为pe增加一个批次维度，使其能够广播到 (batch_size, seq_len, d_model)
        pe = pe.unsqueeze(0)  # 形状: (1, seq_len, d_model)

        # 使用 register_buffer 将 pe 注册为模型的缓冲区。
        # 这意味着 pe 是模型状态的一部分，会随模型移动（如 .to(device)），但不会被视为可训练参数。
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，将位置编码添加到输入嵌入中。

        Args:
            x (torch.Tensor): 输入的嵌入向量，形状为 (batch_size, seq_len, d_model)。

        Returns:
            torch.Tensor: 添加了位置信息并应用了dropout的张量。
        """
        # x 的形状是 (batch, seq_len, d_model)
        # self.pe[:, :x.shape[1], :] 会截取与输入序列长度相匹配的位置编码
        # requires_grad_(False) 确保位置编码本身不是可学习的
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    层归一化 (Layer Normalization)

    对每个样本在特征维度上进行归一化，使其均值为0，方差为1。
    有助于稳定训练过程，加速收敛，并减弱对初始化方法的敏感性。
    """

    def __init__(self, features: int, eps: float = 1e-6):
        """
        初始化层归一化。

        Args:
            features (int): 特征维度的大小 (即 d_model)。
            eps (float): 为防止除以零而添加的一个小值。
        """
        super().__init__()
        self.eps = eps
        # alpha 是可学习的增益参数 (gain)
        self.alpha = nn.Parameter(torch.ones(features))
        # bias 是可学习的偏置参数 (bias)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, features)。

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同。
        """
        # 沿着最后一个维度（特征维度）计算均值和标准差
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # 应用归一化，并通过可学习参数 alpha 和 bias 进行仿射变换
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)

    这是Transformer块中的第二个主要子层。它是一个简单的两层全连接网络，
    对每个位置的表示进行独立的非线性变换。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        初始化前馈网络。

        Args:
            d_model (int): 输入和输出的维度。
            d_ff (int): 中间隐藏层的维度，通常是 d_model 的4倍。
            dropout (float): Dropout的比率。
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # 扩展层
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # 压缩层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        公式: FFN(x) = (Dropout(max(0, xW1 + b1)))W2 + b2

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)

    模型的核心。它允许模型在不同位置的表示子空间中共同关注来自不同位置的信息。
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        """
        初始化多头注意力块。

        Args:
            d_model (int): 模型的维度。
            h (int): 注意力头的数量。
            dropout (float): Dropout的比率。
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        # 确保 d_model 可以被 h 整除
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # 每个头的维度
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # 查询(Query)权重矩阵
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # 键(Key)权重矩阵
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # 值(Value)权重矩阵
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 输出权重矩阵
        self.dropout = nn.Dropout(dropout)
        self.attention_scores = None  # 用于可视化

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ):
        """
        计算缩放点积注意力。

        Args:
            query (torch.Tensor): 查询, 形状 (batch, h, seq_len, d_k)
            key (torch.Tensor): 键, 形状 (batch, h, seq_len, d_k)
            value (torch.Tensor): 值, 形状 (batch, h, seq_len, d_k)
            mask (torch.Tensor): 掩码, 用于屏蔽不需要关注的位置。
            dropout (nn.Dropout): Dropout层。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (注意力输出, 注意力分数)
        """
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # 这里的缩放 ( / math.sqrt(d_k) ) 至关重要。
        # 它可以防止点积结果过大，从而避免softmax函数进入梯度饱和区，稳定训练。

        # mask是一个0 1 矩阵
        if mask is not None:
            # 将掩码中为0的位置填充为一个非常小的负数，这样在softmax后这些位置的概率会接近0。
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # 将 d_model 维度拆分为 h 个头
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # 计算注意力
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # 将多头的结果拼接回来
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # 应用最终的线性变换
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    残差连接与层归一化 (Pre-LN)

    该模块实现了 "Add & Norm" 操作。此处的实现遵循 Pre-LN (Layer Normalization First) 结构，
    即先进行层归一化，再将输入传递给子层（如多头注意力或前馈网络），最后进行残差连接。
    Pre-LN 被认为比原始的 Post-LN 结构训练更稳定。
    """

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            sublayer (nn.Module): 要应用残差连接的子层 (例如 MHA 或 FFN)。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 结构: x + Dropout(Sublayer(LayerNorm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    编码器块 (Encoder Block)

    一个编码器块由两个子层组成：一个多头自注意力层和一个前馈神经网络。
    每个子层都包裹在残差连接中。
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 来自上一层的输入，形状 (batch, seq_len, d_model)。
            src_mask (torch.Tensor): 源序列的掩码，用于屏蔽padding token。

        Returns:
            torch.Tensor: 编码器块的输出，形状与输入相同。
        """
        # 第一个残差连接：多头自注意力
        # lambda x: ... 创建一个匿名函数，使其符合 ResidualConnection 的 sublayer 输入格式
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        # 第二个残差连接：前馈网络
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class DecoderBlock(nn.Module):
    """
    解码器块 (Decoder Block)

    一个解码器块由三个子层组成：一个带掩码的多头自注意力层，一个多头交叉注意力层
    （处理编码器的输出），以及一个前馈神经网络。
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 来自上一解码器层的输入。
            encoder_output (torch.Tensor): 编码器的最终输出。
            src_mask (torch.Tensor): 源序列掩码，用于交叉注意力。
            tgt_mask (torch.Tensor): 目标序列掩码，用于自注意力，防止关注未来token。

        Returns:
            torch.Tensor: 解码器块的输出。
        """
        # 第一个残差连接：带掩码的多头自注意力
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # 第二个残差连接：交叉注意力
        # Q 来自解码器，K 和 V 来自编码器
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        # 第三个残差连接：前馈网络
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    编码器 (Encoder)

    由N个相同的编码器块堆叠而成。
    """

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        # 最后的层归一化（Pre-LN架构的标准做法）
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    解码器 (Decoder)

    由N个相同的解码器块堆叠而成。
    """

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        # 最后的层归一化（Pre-LN架构的标准做法）
        self.norm = LayerNormalization(features)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    投影层 (Projection Layer)

    将解码器的输出向量投影到词汇表空间，生成每个词元的对数概率（logits）。
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    """
    完整的Transformer模型

    整合了编码器、解码器和所有必要的嵌入、位置编码和投影层。
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        对源序列进行编码。

        Args:
            src (torch.Tensor): 源序列的token ID。
            src_mask (torch.Tensor): 源序列的掩码。

        Returns:
            torch.Tensor: 编码后的表示。
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据编码器输出和目标序列进行解码。

        Args:
            encoder_output (torch.Tensor): 编码器的输出。
            src_mask (torch.Tensor): 源序列掩码。
            tgt (torch.Tensor): 目标序列的token ID。
            tgt_mask (torch.Tensor): 目标序列掩码。

        Returns:
            torch.Tensor: 解码后的表示。
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        将解码器输出投影到词汇表空间。

        Args:
            x (torch.Tensor): 解码器的最终输出。

        Returns:
            torch.Tensor: 词汇表空间的logits。
        """
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    工厂函数，用于构建和初始化一个完整的Transformer模型。

    Args:
        src_vocab_size (int): 源语言词汇表大小。
        tgt_vocab_size (int): 目标语言词汇表大小。
        src_seq_len (int): 源序列最大长度。
        tgt_seq_len (int): 目标序列最大长度。
        d_model (int): 模型维度。
        N (int): 编码器和解码器块的数量。
        h (int): 注意力头的数量。
        dropout (float): Dropout比率。
        d_ff (int): 前馈网络隐藏层维度。

    Returns:
        Transformer: 一个完整的、已初始化的Transformer模型。
    """
    # 创建词嵌入层
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # 创建位置编码层
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 创建编码器块列表
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # 创建解码器块列表
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # 创建编码器和解码器
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # 创建投影层
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # 创建Transformer模型
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # 初始化参数
    # Xavier初始化是一种常用的权重初始化方法，有助于在训练开始时保持梯度的稳定。
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
