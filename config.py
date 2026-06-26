from dataclasses import dataclass
import os


@dataclass
class Config:
    #路径配置
    data_path: str = os.path.join(os.path.dirname(__file__), "xiaohuangji50w_fenciA.conv")
    vocab_path: str = os.path.join(os.path.dirname(__file__), "vocab.json")
    model_save_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_path: str = os.path.join(os.path.dirname(__file__), "train.log")

    #词汇表配置
    vocab_size: int = 50000        # 词汇表大小
    min_freq: int = 3              # 最低词频阈值

    #模型架构参数
    d_model: int = 128           # 词向量 / 隐层维度
    n_heads: int = 8               # 多头注意力头数
    n_layers: int = 6              # Encoder / Decoder 层数
    d_ff: int = 2048               # 前馈网络隐层维度
    dropout: float = 0.1           # Dropout 比例
    max_len: int = 60              # 最大序列长度（已分词 token 数）

    #训练参数
    batch_size: int = 128          # 批次大小 (512 → 128, 避免 OOM)
    epochs: int = 30               # 训练轮数
    warmup_steps: int = 4000       # Noam 调度器预热步数
    label_smoothing: float = 0.1   # 标签平滑
    grad_clip: float = 1.0         # 梯度裁剪阈值
    log_every: int = 100           # 每 N 步打印日志
    save_every_epoch: int = 1      # 每 N 个 epoch 保存一次

    #推理参数
    beam_size: int = 5             # Beam Search 宽度
    max_decode_len: int = 50       # 最大解码长度
    temperature: float = 0.8       # 采样温度
    length_penalty: float = 0.6    # 长度惩罚系数

    # 硬件
    device: str = "cuda"           # "cuda" / "cpu"


# 特殊 Token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
