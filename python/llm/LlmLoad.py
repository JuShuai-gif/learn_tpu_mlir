# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from safetensors import safe_open
import os, torch


class LlmLoad:
    """
    用于加载和读取大型语言模型(LLM)的权重文件。
    支持 .safetensors 和 .bin 格式的文件。
    """
    def __init__(self, model_path: str):
        """
        初始化加载器。
        :param model_path: 模型权重文件所在的文件夹路径。
        """
        self.st_files = []# 存储已打开的权重文件句柄（可能是safetensors或普通torch权重）
        # get all safetensors
        # 遍历指定文件夹下所有文件
        for entry in os.listdir(model_path):
            file_path = os.path.join(model_path, entry)# 拼接完整文件路径
            if os.path.isfile(file_path):# 确保是文件（不是目录）
                # 处理 safetensors 文件
                if entry.lower().endswith('.safetensors'):
                    # safetensors 提供了安全的、只读的张量读取方式
                    f = safe_open(file_path, "pt")# "pt" 表示 PyTorch 格式
                    self.st_files.append(f)
                # 处理 PyTorch 传统 .bin 文件
                elif entry.lower().endswith('.bin'):
                    # 直接用 torch.load 加载，放到 CPU 上
                    f = torch.load(file_path, map_location="cpu")
                    self.st_files.append(f)

    def read(self, key: str):
        """
        读取指定 key 对应的权重张量，并返回 numpy 数组格式。
        如果 key 对应的数据是半精度（float16）或 bfloat16，会转换成 float32。
        :param key: 权重名称（例如 "model.layers.0.self_attn.q_proj.weight"）
        :return: 对应的 numpy 数组
        :raises RuntimeError: 如果没有找到对应 key
        """
        for f in self.st_files:
            # safetensors 句柄有 keys() 方法，torch.load 的结果是字典
            if key in f.keys():
                # 如果是普通字典（torch.load 加载的 .bin）
                if isinstance(f, dict):
                    data = f[key]
                else:
                    # safetensors 用 get_tensor 方法读取
                    data = f.get_tensor(key)
                # 半精度和bfloat16类型转换为float32
                if data.dtype in [torch.float16, torch.bfloat16]:
                    return data.float().numpy()
                # 其他类型直接返回 numpy
                return data.numpy()
        # 如果所有文件中都找不到该 key，则抛出异常
        raise RuntimeError(f"Can't find key: {key}")

    def is_exist(self, key: str):
        """
        检查是否存在指定 key 的权重。
        :param key: 权重名称
        :return: True(存在) / False(不存在)
        """
        for f in self.st_files:
            if key in f.keys():
                return True
        return False
