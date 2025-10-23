#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据集加载脚本
用于验证opus100英中数据集的加载和格式
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import torch
from config import get_config

def test_dataset_loading():
    """测试数据集加载"""
    print("🧪 开始测试opus100英译汉数据集加载...")
    config = get_config()

    try:
        # 加载数据集 - 现在是en-zh方向
        print("📥 正在加载opus100 en-zh数据集...")
        ds_full = load_dataset("opus100", "en-zh")

        print("✅ 数据集加载成功!")
        print(f"数据集包含的子集: {list(ds_full.keys())}")
        print(f"训练集大小: {len(ds_full['train'])} samples")
        print(f"验证集大小: {len(ds_full['validation'])} samples")
        print(f"测试集大小: {len(ds_full['test'])} samples")

        # 检查数据格式
        print("\n🔍 检查数据格式:")
        train_sample = ds_full['train'][0]
        print(f"第一条训练数据: {train_sample}")

        # 验证翻译字段
        if 'translation' in train_sample:
            translation = train_sample['translation']
            print(f"\n翻译字段内容:")
            print(f"  英文字段存在: {'en' in translation}")
            print(f"  中文字段存在: {'zh' in translation}")

            if 'en' in translation and 'zh' in translation:
                print(f"\n📄 样本示例:")
                print(f"  英文: {translation['en']}")
                print(f"  中文: {translation['zh']}")

                # 检查字符长度
                en_len = len(translation['en'])
                zh_len = len(translation['zh'])
                print(f"\n📏 长度对比:")
                print(f"  英文长度: {en_len} 字符")
                print(f"  中文长度: {zh_len} 字符")
                print(f"  英中长度比: {en_len/zh_len:.2f}")

        # 检查几条随机样本
        print(f"\n🎯 更多随机样本 (英文→中文):")
        for i in range(3):
            sample = ds_full['train'][i]
            en_text = sample['translation']['en']
            zh_text = sample['translation']['zh']
            print(f"  样本{i+1}: '{en_text[:50]}...' -> '{zh_text[:50]}...'")

        print(f"\n🔧 配置验证:")
        print(f"  源语言: {config['lang_src']}")
        print(f"  目标语言: {config['lang_tgt']}")
        print(f"  数据方向: {config['datasource']} en-zh")

        print("\n✨ 数据集测试完成！格式正确，可以开始英译汉训练。")
        return True

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False

if __name__ == "__main__":
    test_dataset_loading()