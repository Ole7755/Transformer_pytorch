# config.py
#
# 该文件负责管理项目的所有配置项，包括文件路径、模型超参数、训练参数等。
# 使用一个专门的函数来返回配置字典，可以方便地在其他文件中导入和使用，
# 也有利于后续通过命令行参数或配置文件（如YAML）来覆盖默认配置。

from pathlib import Path


def get_config():
    """
    返回一个包含所有配置的字典。
    使用Pathlib处理路径，增强跨平台兼容性。
    """
    return {
        "batch_size": 20,
        "num_epochs": 1,
        "lr": 1e-4,
        "seq_len": 400,  # 保持400长度，中英文都适用
        "d_model": 512,  # 模型的维度
        "lang_src": "en",  # 源语言：英文（修改回英文）
        "lang_tgt": "zh",  # 目标语言：中文（修改为目标语言）
        "model_folder": "weights",  # 模型权重保存文件夹
        "model_basename": "en2zh_model_",  # 更名为英中模型
        "preload": "latest",  # 预加载模型权重, "latest"表示加载最新的
        "tokenizer_file": "tokenizer_{0}.json",  # Tokenizer保存路径模板
        "experiment_name": "runs/en2zh_model",  # TensorBoard实验名称
        "datasource": "opus100",  # 数据集名称：opus100英中平行语料
    }


def get_weights_file_path(config, epoch: str):
    """
    根据epoch号构造模型权重文件的完整路径。

    Args:
        config (dict): 配置字典。
        epoch (str): epoch号，可以是数字或"latest"。

    Returns:
        str: 模型权重文件的路径。
    """
    model_folder = Path(config["model_folder"])
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(model_folder / model_filename)
