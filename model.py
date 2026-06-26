import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, PAD_ID



# 位置编码

class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)
        返回: (B, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)



#多头注意力
class MultiHeadAttention(nn.Module):
    """多头缩放点积注意力。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, d_model) → (B, n_heads, seq_len, d_k)"""
        B, seq_len, _ = x.shape
        x = x.view(B, seq_len, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (B, n_heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, seq_len, d_k) → (B, seq_len, d_model)"""
        B, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, seq_len, n_heads, d_k)
        return x.view(B, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query/key/value: (B, seq_len, d_model)
        mask: (B, 1, 1, key_len) 或 (B, 1, q_len, key_len) — True = 遮掩
        返回: (B, q_len, d_model)
        """
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 拆分为多头
        Q = self._split_heads(Q)  # (B, n_heads, q_len, d_k)
        K = self._split_heads(K)  # (B, n_heads, k_len, d_k)
        V = self._split_heads(V)  # (B, n_heads, k_len, d_k)

        # 注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, q_len, k_len)

        if mask is not None:
            # mask: (B, 1, q_len, k_len) 或 (B, 1, 1, k_len)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)  # (B, n_heads, q_len, d_k)
        output = self._merge_heads(output)       # (B, q_len, d_model)
        return self.w_o(output)



#  前馈网络
class FeedForward(nn.Module):
    """两层全连接 + ReLU 激活。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


#  Encoder


class EncoderLayer(nn.Module):
    """单层 Encoder: Self-Attention → FFN，使用 Pre-LN。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, src_len, d_model)
        mask: (B, 1, 1, src_len) — padding mask
        """
        # Self-Attention (Pre-LN)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class Encoder(nn.Module):
    """Encoder = 词嵌入 + 位置编码 + N 层 EncoderLayer。"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # 最终 LayerNorm

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, src_len) — token IDs
        mask: (B, 1, 1, src_len) — padding mask (True = padding)
        返回: (B, src_len, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x



#  Decoder


class DecoderLayer(nn.Module):
    """单层 Decoder: Masked Self-Attention → Cross-Attention → FFN，使用 Pre-LN。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, tgt_len, d_model)
        enc_output: (B, src_len, d_model)
        self_mask: (B, 1, tgt_len, tgt_len) — causal + padding mask
        cross_mask: (B, 1, 1, src_len) — source padding mask
        """
        # Masked Self-Attention (Pre-LN)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, self_mask)
        x = self.dropout(x)
        x = residual + x

        # Cross-Attention (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.dropout(x)
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class Decoder(nn.Module):
    """Decoder = 词嵌入 + 位置编码 + N 层 DecoderLayer。"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, tgt_len) — token IDs
        enc_output: (B, src_len, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)

        x = self.norm(x)
        return x



# Transformer


class Transformer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
        )

        # 输出投影
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # 权重共享：Encoder 嵌入、Decoder 嵌入、输出投影共享权重矩阵
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.output_proj.weight = self.encoder.embedding.weight

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """Xavier / 正态初始化。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_padding_mask(pad_mask: torch.Tensor) -> torch.Tensor:
        """
        将 (B, seq_len) 的布尔 padding mask 转为注意力可用的形状。

        pad_mask: (B, seq_len) — True = padding
        返回: (B, 1, 1, seq_len) — True = 遮掩
        """
        # (B, seq_len) → (B, 1, 1, seq_len)
        return pad_mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果 mask（上三角为 True）。
        返回: (1, 1, seq_len, seq_len)
        """
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        src: (B, src_len) — 源序列 token IDs
        tgt_input: (B, tgt_len) — 目标序列 token IDs（含 <SOS>）
        src_pad_mask: (B, src_len) — True = padding
        tgt_pad_mask: (B, tgt_len) — True = padding

        返回: (B, tgt_len, vocab_size) — logits
        """
        # 构造注意力 Mask
        enc_mask = self.make_padding_mask(src_pad_mask)          # (B, 1, 1, src_len)

        tgt_len = tgt_input.size(1)
        causal = self.make_causal_mask(tgt_len, tgt_input.device)  # (1, 1, tgt_len, tgt_len)
        tgt_padding = tgt_pad_mask.unsqueeze(1).unsqueeze(2)       # (B, 1, 1, tgt_len)
        dec_self_mask = causal | tgt_padding                        # (B, 1, tgt_len, tgt_len)
        dec_cross_mask = self.make_padding_mask(src_pad_mask)       # (B, 1, 1, src_len)

        # Encoder
        enc_output = self.encoder(src, enc_mask)  # (B, src_len, d_model)

        # Decoder
        dec_output = self.decoder(tgt_input, enc_output, dec_self_mask, dec_cross_mask)  # (B, tgt_len, d_model)

        # 输出投影
        logits = self.output_proj(dec_output)  # (B, tgt_len, vocab_size)

        return logits

    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        """单独调用 Encoder（推理时复用编码结果）。"""
        enc_mask = self.make_padding_mask(src_pad_mask)
        return self.encoder(src, enc_mask)

    def decode_step(
        self,
        tgt_token: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        cross_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步解码。

        tgt_token: (B, 1) — 当前步的 token ID
        enc_output: (B, src_len, d_model)
        返回: (B, 1, vocab_size) — logits
        """
        tgt_len = tgt_token.size(1)
        causal = self.make_causal_mask(tgt_len, tgt_token.device)  # (1, 1, 1, 1) — 全 0
        tgt_padding = tgt_pad_mask.unsqueeze(1).unsqueeze(2)
        dec_self_mask = causal | tgt_padding
        dec_cross_mask = self.make_padding_mask(cross_pad_mask)

        dec_output = self.decoder(tgt_token, enc_output, dec_self_mask, dec_cross_mask)
        return self.output_proj(dec_output)


#  模型测试
if __name__ == "__main__":
    config = Config()
    model = Transformer(config)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量:   {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 测试前向传播
    B, src_len, tgt_len = 4, 20, 18
    src = torch.randint(4, config.vocab_size, (B, src_len))
    tgt_input = torch.randint(4, config.vocab_size, (B, tgt_len))
    src_pad_mask = torch.zeros(B, src_len, dtype=torch.bool)
    src_pad_mask[:, -2:] = True  # 模拟 padding
    tgt_pad_mask = torch.zeros(B, tgt_len, dtype=torch.bool)

    logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask)
    print(f"\n输入  src:       {src.shape}")
    print(f"输入  tgt_input: {tgt_input.shape}")
    print(f"输出  logits:    {logits.shape}")  # 期望: (4, 18, 50000)

    # 验证输出形状
    assert logits.shape == (B, tgt_len, config.vocab_size), f"形状错误: {logits.shape}"
    print("\n✓ 模型前向传播测试通过！")
