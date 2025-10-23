#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集长度分析工具
分析opus100数据集中句子的长度分布，帮助确定合适的序列长度
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import numpy as np
from config import get_config

def analyze_sentence_lengths(max_samples=10000):
    """分析数据集中句子的长度分布"""
    print("🔍 开始分析opus100数据集句子长度分布...")
    config = get_config()

    try:
        # 加载数据集
        print("📥 加载opus100 en-zh数据集...")
        ds_full = load_dataset("opus100", "en-zh")

        # 分析训练集
        train_ds = ds_full['train']
        print(f"训练集总大小: {len(train_ds)} 样本")

        # 如果数据集太大，采样分析
        sample_size = min(max_samples, len(train_ds))
        indices = np.random.choice(len(train_ds), sample_size, replace=False)

        en_lengths = []
        zh_lengths = []
        total_pairs = 0

        print(f"分析前{sample_size}个样本的长度分布...")

        for idx in indices:
            sample = train_ds[int(idx)]
            en_text = sample['translation']['en']
            zh_text = sample['translation']['zh']

            # 英文字符长度
            en_len = len(en_text)
            # 中文字符长度
            zh_len = len(zh_text)

            en_lengths.append(en_len)
            zh_lengths.append(zh_len)
            total_pairs += 1

            if total_pairs % 2000 == 0:
                print(f"  已分析 {total_pairs} 个样本...")

        # 转换为numpy数组便于分析
        en_lengths = np.array(en_lengths)
        zh_lengths = np.array(zh_lengths)

        print(f"\n📊 数据分析结果 (基于{total_pairs}个样本):")
        print("="*60)

        # 英文长度统计
        print("🔤 英文句子长度统计:")
        print(f"  平均长度: {np.mean(en_lengths):.1f} 字符")
        print(f"  中位数: {np.median(en_lengths):.1f} 字符")
        print(f"  标准差: {np.std(en_lengths):.1f} 字符")
        print(f"  最小长度: {np.min(en_lengths)} 字符")
        print(f"  最大长度: {np.max(en_lengths)} 字符")
        print(f"  95%分位数: {np.percentile(en_lengths, 95):.1f} 字符")
        print(f"  99%分位数: {np.percentile(en_lengths, 99):.1f} 字符")

        # 中文长度统计
        print("\n🇨🇳 中文句子长度统计:")
        print(f"  平均长度: {np.mean(zh_lengths):.1f} 字符")
        print(f"  中位数: {np.median(zh_lengths):.1f} 字符")
        print(f"  标准差: {np.std(zh_lengths):.1f} 字符")
        print(f"  最小长度: {np.min(zh_lengths)} 字符")
        print(f"  最大长度: {np.max(zh_lengths)} 字符")
        print(f"  95%分位数: {np.percentile(zh_lengths, 95):.1f} 字符")
        print(f"  99%分位数: {np.percentile(zh_lengths, 99):.1f} 字符")

        # 计算分词后的长度（考虑特殊标记）
        print(f"\n🧮 考虑到分词后的长度预估:")
        # 英文：平均每个单词约5字符，加上空格
        en_tokenized = en_lengths * 1.2  # 粗略估计
        # 中文：字符级分词，基本保持原长度，加上特殊标记
        zh_tokenized = zh_lengths * 1.1 + 2  # + SOS和EOS

        print(f"  英文分词后平均长度: {np.mean(en_tokenized):.1f}")
        print(f"  中文分词后平均长度: {np.mean(zh_tokenized):.1f}")

        # 计算cover率
        current_seq_len = config['seq_len']
        en_cover_95 = np.percentile(en_tokenized, 95)
        zh_cover_95 = np.percentile(zh_tokenized, 95)
        max_cover_95 = max(en_cover_95, zh_cover_95)

        print(f"\n📏 与当前序列长度({current_seq_len})的对比:")
        print(f"  95%的英文句子分词后 < {en_cover_95:.1f}")
        print(f"  95%的中文句子分词后 < {zh_cover_95:.1f}")
        print(f"  建议序列长度(覆盖95%): {max_cover_95:.0f}")
        print(f"  建议序列长度(覆盖99%): {max(np.percentile(en_tokenized, 99), np.percentile(zh_tokenized, 99)):.0f}")

        # 超限统计
        en_over_limit = np.sum(en_tokenized > current_seq_len)
        zh_over_limit = np.sum(zh_tokenized > current_seq_len)

        print(f"\n⚠️  超限情况分析:")
        print(f"  英文超过{current_seq_len}的样本: {en_over_limit}个 ({en_over_limit/total_pairs*100:.2f}%)")
        print(f"  中文超过{current_seq_len}的样本: {zh_over_limit}个 ({zh_over_limit/total_pairs*100:.2f}%)")

        return {
            'en_stats': {
                'mean': np.mean(en_lengths),
                'median': np.median(en_lengths),
                'max': np.max(en_lengths),
                'p95': np.percentile(en_lengths, 95),
                'p99': np.percentile(en_lengths, 99)
            },
            'zh_stats': {
                'mean': np.mean(zh_lengths),
                'median': np.median(zh_lengths),
                'max': np.max(zh_lengths),
                'p95': np.percentile(zh_lengths, 95),
                'p99': np.percentile(zh_lengths, 99)
            },
            'recommendations': {
                'seq_len_95': max_cover_95,
                'seq_len_99': max(np.percentile(en_tokenized, 99), np.percentile(zh_tokenized, 99))
            }
        }

    except Exception as e:
        print(f"❌ 数据分析失败: {e}")
        return None

if __name__ == "__main__":
    result = analyze_sentence_lengths(max_samples=5000)  # 分析5000个样本
    if result:
        print(f"\n✅ 分析完成！建议的序列长度: {result['recommendations']['seq_len_95']:.0f}")
    else:
        print("""\n💡 临时建议:
1. 将序列长度调整为512或600（覆盖95%的数据）
2. 或者添加数据截断/过滤逻辑
3. 对于特别长的句子，可以考虑截断而不是完全丢弃""")