# train.py
#
# 这是项目的主执行文件，负责整个训练流程的控制。
# 它整合了config, dataset, model三个模块，执行以下关键步骤：
# 1. 加载和预处理数据。
# 2. 构建或加载分词器（Tokenizer）。
# 3. 初始化Transformer模型。
# 4. 设置优化器和损失函数。
# 5. 运行训练循环（training loop），包括：
#    - 前向传播
#    - 计算损失
#    - 反向传播和参数更新
# 6. 定期进行模型验证和保存。
# 7. 使用TensorBoard进行训练过程的可视化。

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tqdm import tqdm
import warnings
from pathlib import Path

# 导入自定义模块
from model import build_transformer
from dataset import CausalLanguageModelingDataset
from config import get_config, get_weights_file_path

# Hugging Face的tokenizers和datasets库
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset

def get_all_sentences(ds, lang):
    """从数据集中提取特定语言的所有句子"""
    for item in ds:
        yield item['translation'][lang]

def chinese_character_tokenizer(text):
    """
    中文字符级分词器
    将中文文本按字符分割，每个字符作为一个token
    """
    tokens = []
    for char in text:
        if char.strip():  # 跳过空白字符
            tokens.append(char)
    return tokens

def get_or_build_tokenizer(config, ds, lang):
    """
    加载或构建分词器。
    如果tokenizer文件已存在，则直接加载；否则，基于数据集训练一个新的tokenizer。
    针对英文和中文采用不同的分词策略（英→中翻译方向）。
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        print(f"Tokenizer for {lang} not found. Building...")

        if lang == 'en':
            # 英文：保持按空格分词（源语言）
            print(f"Building word-level tokenizer for English...")
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        elif lang == 'zh':
            # 中文：字符级分词（目标语言）
            print(f"Building character-level tokenizer for Chinese...")
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

            # 收集所有中文字符作为词汇
            vocab_set = set()
            for item in ds:
                text = item['translation'][lang]
                for char in text:
                    if char.strip() and char not in ['[UNK]', '[PAD]', '[SOS]', '[EOS]']:
                        vocab_set.add(char)

            # 创建词汇表
            vocab_list = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"] + list(vocab_set)
            vocab = {token: idx for idx, token in enumerate(vocab_list)}

            # 重新初始化tokenizer以使用我们的词汇表
            from tokenizers.models import WordLevel as WordLevelModel
            model = WordLevelModel(vocab=vocab, unk_token="[UNK]")
            tokenizer = Tokenizer(model)

        else:
            raise ValueError(f"Unsupported language: {lang}")

        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Loading tokenizer for {lang}...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    """加载数据集并进行预处理"""
    # 从Hugging Face加载opus100英中平行语料(en-zh)
    print(f"Loading opus100 English-Chinese dataset...")
    ds_full = load_dataset("opus100", "en-zh")

    # opus100已经预先分割好了train/validation/test，我们使用train和validation部分
    ds_train = ds_full['train']
    ds_validation = ds_full['validation']

    print(f"Train dataset size: {len(ds_train)}")
    print(f"Validation dataset size: {len(ds_validation)}")

    # 显示几个样本以验证数据格式(现在是en→zh方向)
    print("\nDataset sample (English → Chinese):")
    for i in range(3):
        sample = ds_train[i]
        print(f"Sample {i+1}: {sample['translation']['en']} -> {sample['translation']['zh']}")

    # 构建分词器 - 使用训练集构建分词器
    print(f"\nBuilding tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, ds_train, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_train, config['lang_tgt'])

    # 直接使用opus100的验证集作为验证数据
    train_ds = CausalLanguageModelingDataset(ds_train, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = CausalLanguageModelingDataset(ds_validation, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # 打印词汇表大小
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")

    # 创建DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False) # 验证时batch_size为1

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """构建Transformer模型"""
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], d_model=config['d_model'])
    return model

def train_model(config):
    """完整的模型训练函数"""
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

    # 确保权重文件夹存在
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # 获取数据
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # 获取模型
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # 设置TensorBoard
    writer = SummaryWriter(config['experiment_name'])

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 加载预训练权重（如果配置了）
    initial_epoch = 0
    global_step = 0
    if config['preload'] == 'latest':
        model_filename = get_weights_file_path(config, "latest")
        if Path(model_filename).exists():
            print(f"Preloading model {model_filename}")
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
            print("No model to preload, starting from scratch.")


    # 设置损失函数，忽略pad_token
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # 开始训练循环
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # 运行Tensors通过Transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # 获取标签
            label = batch['label'].to(device) # (B, seq_len)

            # 计算损失
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # 记录到TensorBoard
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # 在每个epoch后运行验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)

                label = batch['label'].to(device)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        writer.add_scalar('val_loss', val_loss, global_step)
        writer.flush()
        print(f"Validation loss for epoch {epoch}: {val_loss}")


        # 保存模型权重
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        # 更新latest模型
        latest_filename = get_weights_file_path(config, "latest")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, latest_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)