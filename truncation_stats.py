#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截断统计工具
监控和报告数据截断情况
"""

import torch
from dataset import CausalLanguageModelingDataset

class TruncationStats:
    """截断统计类，用于监控数据截断情况"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.total_samples = 0
        self.src_truncated = 0
        self.tgt_truncated = 0

    def update(self, original_src_tokens, truncated_src_tokens, original_tgt_tokens, truncated_tgt_tokens):
        """更新截断统计"""
        self.total_samples += 1
        if len(original_src_tokens) != len(truncated_src_tokens):
            self.src_truncated += 1
        if len(original_tgt_tokens) != len(truncated_tgt_tokens):
            self.tgt_truncated += 1

    def get_stats(self):
        """获取统计结果"""
        if self.total_samples == 0:
            return {
                'total': 0,
                'src_truncated': 0,
                'tgt_truncated': 0,
                'src_ratio': 0.0,
                'tgt_ratio': 0.0
            }

        return {
            'total': self.total_samples,
            'src_truncated': self.src_truncated,
            'tgt_truncated': self.tgt_truncated,
            'src_ratio': self.src_truncated / self.total_samples * 100,
            'tgt_ratio': self.tgt_truncated / self.total_samples * 100
        }

    def print_stats(self):
        """打印统计结果"""
        stats = self.get_stats()
        print(f"📊 截断统计报告:")
        print(f"  总样本数: {stats['total']}")
        print(f"  源语言截断数: {stats['src_truncated']} ({stats['src_ratio']:.2f}%)")
        print(f"  目标语言截断数: {stats['tgt_truncated']} ({stats['tgt_ratio']:.2f}%)")
        print(f"  整体截断率: {(stats['src_ratio'] + stats['tgt_ratio']):.2f}%")