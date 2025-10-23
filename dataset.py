# dataset.py
#
# 该文件定义了数据加载和预处理的逻辑。
# 核心是CausalLanguageModelingDataset类，它继承自PyTorch的Dataset类，
# 负责处理原始文本数据，通过Tokenizer进行分词，并构建模型所需的输入张量。

import torch
from torch.utils.data import Dataset

class CausalLanguageModelingDataset(Dataset):
    """
    自定义数据集，用于因果语言模型任务。
    这个数据集负责将成对的文本（源语言和目标语言）转换成模型训练所需的格式。
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        """
        初始化数据集。

        Args:
            ds (Dataset): 原始Hugging Face数据集。
            tokenizer_src (Tokenizer): 源语言的分词器。
            tokenizer_tgt (Tokenizer): 目标语言的分词器。
            src_lang (str): 源语言标识符 (e.g., 'en')。
            tgt_lang (str): 目标语言标识符 (e.g., 'it')。
            seq_len (int): 序列的最大长度。
        """
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # 获取特殊标记的ID
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """
        根据索引获取一个数据样本，添加智能截断逻辑避免序列长度超限错误

        返回一个字典，包含编码器输入、解码器输入、各种掩码和标签
        """
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # 对源文本和目标文本进行分词
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 智能截断逻辑：为特殊标记预留空间
        max_enc_length = self.seq_len - 2  # 为SOS和EOS预留空间
        max_dec_length = self.seq_len - 1  # 为SOS预留空间

        # 如果序列过长，智能截断保留重要信息
        if len(enc_input_tokens) > max_enc_length:
            # 优先保留句子后面的重要部分，截断前面的辅助信息
            enc_input_tokens = enc_input_tokens[-max_enc_length:]

        if len(dec_input_tokens) > max_dec_length:
            # 智能截断中文目标序列
            dec_input_tokens = dec_input_tokens[-max_dec_length:]

        # 计算需要填充的长度（现在是安全运算，不会为负）
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # 构建编码器输入 (encoder_input)
        # 格式: [SOS, src_tokens, EOS, PAD, ...]
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 构建解码器输入 (decoder_input)
        # 格式: [SOS, tgt_tokens, PAD, ...]
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 构建标签 (label)
        # 格式: [tgt_tokens, EOS, PAD, ...]
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 智能处理截断统计
        if hasattr(self, '_truncation_stats'):
            self._truncation_stats['total_samples'] += 1
            if enc_input_tokens != self.tokenizer_src.encode(src_text).ids:
                self._truncation_stats['src_truncated'] += 1
            if dec_input_tokens != self.tokenizer_tgt.encode(tgt_text).ids:
                self._truncation_stats['tgt_truncated'] += 1

        # 确保所有张量的长度都等于seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # 返回模型需要的所有数据
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & self.create_causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
    @staticmethod
    def create_causal_mask(size):
        # 创建一个下三角矩阵作为因果掩码，防止解码器看到未来的信息
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0