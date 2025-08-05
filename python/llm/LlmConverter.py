# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torch
import os
from transform.MLIRImporter import MLIRImporter, Platform
from transform.BaseConverter import BaseConverter
from .LlmInfo import *
from .LlmLoad import *
import numpy as np
from tqdm import tqdm
from datetime import datetime

import concurrent.futures
import subprocess
import sys
from mlir.ir import *
import mlir.dialects.top as top


class LlmConverter(BaseConverter):

    def __init__(self, args, config):
        super().__init__()
        # 模型路径
        self.model_path = os.path.normpath(args.model_path)
        # 序列长度
        self.seq_length = args.seq_length
        # 最大输入长度
        self.max_input_length = args.max_input_length if (
            args.max_input_length > 0
            and args.max_input_length < self.seq_length) else self.seq_length
        # 最大p kv长度
        self.max_prefill_kv_length = args.max_prefill_kv_length
        # 量化
        self.quantize = args.quantize
        # 设备数量
        self.num_device = args.num_device
        # 量化组数量
        self.q_group_size = args.q_group_size
        # 强制高精度模式
        self.high_precision = True
        # 是否使用对称量化（对称vs非对称）
        self.symmetric = args.symmetric
        # 是否对输出层进行Top-K优化
        self.lmhead_with_topk = True if not args.do_sample else False
        # 目标芯片型号（如bm1684x/bm1688）
        self.chip = args.chip
        # 是否将词嵌入存储到磁盘（省内存）
        self.embedding_disk = args.embedding_disk
        # 是否支持动态输入长度
        self.dynamic = args.dynamic
        # 是否使用KV缓存块加速自回归生成
        self.use_block_with_kv = args.use_block_with_kv
        # 调试模式开关
        self.debug = args.debug
        # 位置编码的 shape
        self.position_shape = [1, self.max_input_length]
        # 计算核心数,bm1688默认2个核
        self.num_core = args.num_core
        if self.num_core == 0:
            self.num_core = 1 if args.chip != "bm1688" else 2
        # 量化精度选择
        self.half_precision_quantize = "bf16" if "bf16" in self.quantize else "f16"
        self.quant_mode = None
        self.quant_bits = 0
        self.vit_f16_out_bf16 = False  # force vit f16, output bf16
        # 初始化配置
        self.load_pretrained(config)
        self.llm_config.max_position_embeddings = self.seq_length
        self.llm_config.rope_scaling = None  # no need rope scaling
        # get attributes
        self.init_config()
        self.do_vit = False
        self.cos, self.sin = self.rotary_embedding()
        cpu_count = os.cpu_count()
        self.max_workers = max(cpu_count, 4)
        # get file path
        self.out_dir = os.path.abspath(args.out_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = os.path.basename(self.model_path).lower()
        if args.chip == "bm1684x":
            folder_name = f"{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_device}dev"
        else:
            folder_name = f"{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_core}core"
        self.out_bmodel = os.path.join(self.out_dir, f"{folder_name}_{timestamp}.bmodel")
        self.bmodel_dir = os.path.join(self.out_dir, folder_name)
        self.config_dir = os.path.join(self.out_dir, "config")
        self.commands = []

    # 运行模型转换流程
    def run(self):
        # 创建输出文件夹(确保目录存在，不存在则创建)
        os.makedirs(self.bmodel_dir, exist_ok=True)
        # 生成模型配置文件(包含量化参数等关键信息)
        self.gen_config()
        # 保存当前工作目录路径，用于后续恢复
        ori_path = os.getcwd()
        # 切换到模型输出目录(所有中间文件将生成在此目录)
        os.chdir(self.bmodel_dir)
        # 生成所有 mlir 中间表示文件(从原始模型转换)
        self.gen_all_mlir()
        # 删除原始模型对象，释放内存
        del self.model
        # 编译所有MLIR文件为目标芯片的bmodel格式
        self.compile_all()
        # 切换回原始工作目录
        os.chdir(ori_path)
        # 输出成功信息（包含原始模型路径和最终输出目录）
        print(f"Success: {self.model_path} has converted to {self.out_dir}")

    def gen_config(self):
        import shutil
        # copy model json file to config dir
        if self.config_dir.startswith(os.path.abspath(self.model_path)):
            os.rmdir(self.bmodel_dir)
            os.rmdir(self.out_dir)
            raise RuntimeError("Can't run under original model path!")
        shutil.copytree(self.model_path,
                        self.config_dir,
                        ignore=shutil.ignore_patterns("*.safetensors", ".*", "*.pth", "*.pt",
                                                      "*.py", "*.bin", "*.bin.index.json",
                                                      "model.safetensors.index.json"),
                        dirs_exist_ok=True)

    def gen_all_mlir(self):
        if self.debug:
            self.gen_vit_mlir()
            self.gen_embedding_lmhead_mlir()
            self.gen_sample_head_mlir()
            for i in range(self.num_layers):
                self.gen_block_mlir(i)
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            if self.do_vit:
                futures.append(executor.submit(self.gen_vit_mlir))

            futures.append(executor.submit(self.gen_embedding_lmhead_mlir))

            if not self.lmhead_with_topk:
                futures.append(executor.submit(self.gen_sample_head_mlir))

            for i in range(self.num_layers):
                futures.append(executor.submit(self.gen_block_mlir, i))

            # Wait for all threads to complete
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="generate mlir"):
                try:
                    # This will raise exceptions if any occurred during thread execution
                    future.result()
                except Exception as e:
                    for future in futures:
                        if not future.done():
                            future.cancel()
                    print(f"Error:gen mlir failed: {e}")
                    sys.exit(1)

    def load_pretrained(self, config):
        self.config = config
        # 加载模型
        self.model = LlmLoad(self.model_path)
        # 模型类型
        self.model_type = self.config.model_type
        # 模型信息
        self.model_info = COMMON_INFO
        # default llm_config is model config; but in vlm, maybe it is not the same
        # 默认 llm_config是模型配置，但是在vlm，可能不一样
        if hasattr(self.config, "text_config"):
            self.llm_config = self.config.text_config
        else:
            self.llm_config = config
        # llm模型类型
        self.llm_type = self.llm_config.model_type

    # rope
    def rotary_embedding(self):
        # 直接使用trnasform的Rope
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        # 创建rope对象
        rotary_embed = LlamaRotaryEmbedding(config=self.llm_config)
        # 位置索引,[1,self.seq_length]
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(1, self.seq_length)
        # 初始化一个 x
        x = torch.zeros([1, self.seq_length, self.hidden_size], dtype=torch.float32)
        # 根据x的shape计算cos、sin
        cos, sin = rotary_embed(x, position_ids)
        #
        cos = cos.reshape(self.seq_length, 1, -1)
        sin = sin.reshape(self.seq_length, 1, -1)
        # 返回numpy格式数据
        return cos.numpy(), sin.numpy()  #[seq, 1, 64]

    def rms_norm(self, mlir_gen, in_op, norm_path: str, name: str = "", eps=None):
        if not self.model.is_exist(norm_path + ".weight"):
            return in_op
        input_shape = list(in_op.type.shape)
        norm_shape = [1] * (len(input_shape) - 1) + [input_shape[-1]]
        # 创建 权重op
        weight_op = mlir_gen.create_weight_op(norm_path + ".weight", norm_shape)
        loc_name = name if name else norm_path
        eps = self.rms_norm_eps if eps is None else eps
        # 返回算子
        return top.RMSNormOp(mlir_gen.get_tensor_type(input_shape),
                             in_op,
                             weight_op,
                             eps=eps,
                             loc=self.get_loc(loc_name, mlir_gen),
                             ip=mlir_gen.insert_point).output

    # layer_norm
    def layer_norm(self, mlir_gen, in_op, norm_path: str, eps, name: str = ""):
        if not self.model.is_exist(norm_path + ".weight"):
            return in_op
        input_shape = list(in_op.type.shape)
        norm_shape = [1] * (len(input_shape) - 1) + [input_shape[-1]]
        # 权重和偏置
        weight_op = mlir_gen.create_weight_op(norm_path + ".weight", norm_shape)
        bias_op = mlir_gen.create_weight_op(norm_path + ".bias", norm_shape)
        loc_name = name if name else norm_path
        # 返回top
        return top.LayerNormOp(mlir_gen.get_tensor_type(input_shape),
                               in_op,
                               weight_op,
                               bias_op,
                               normalized_shape=[input_shape[-1]],
                               axis=len(input_shape) - 1,
                               eps=eps,
                               loc=self.get_loc(loc_name, mlir_gen),
                               ip=mlir_gen.insert_point).output

    # 激活函数
    def activate(self, mlir_gen, in_op, act_type: ActType, path: str):
        input_shape = list(in_op.type.shape)
        if act_type == ActType.SILU:
            return top.SiLUOp(mlir_gen.get_tensor_type(input_shape),
                              in_op,
                              loc=self.get_loc(path + ".silu", mlir_gen),
                              ip=mlir_gen.insert_point).output
        elif act_type == ActType.GELU_PYTORCH_TANH:
            return top.GELUOp(mlir_gen.get_tensor_type(input_shape),
                              in_op,
                              approx_mode=StringAttr.get("tanh"),
                              loc=self.get_loc(path + ".gelu", mlir_gen),
                              ip=mlir_gen.insert_point).output
        elif act_type == ActType.QUICK_GELU:
            return top.SwishOp(mlir_gen.get_tensor_type(input_shape),
                               in_op,
                               beta=1.702,
                               loc=self.get_loc(path + ".swish", mlir_gen),
                               ip=mlir_gen.insert_point).output
        elif act_type == ActType.GELU:
            return top.GELUOp(mlir_gen.get_tensor_type(input_shape),
                              in_op,
                              loc=self.get_loc(path + ".gelu", mlir_gen),
                              ip=mlir_gen.insert_point).output
        else:
            raise NotImplementedError(f"Unsupported activation type: {act_type}")

    # 权重处理
    def unpack_weights(self, qweight, qzeros, bits, quant_mode):
        # 用于存放解压后的权重时的目标数据类型
        dtype = np.int32
        # 压缩率，32位数据中存放多少个量化值
        compress_ratio = 32 // bits
        # 掩码(提取低 bits 位数据时用)
        mask = 0xF if bits == 4 else 0xFF

        # 获取量化权重和零点的维度
        K, N = qweight.shape    # qweight 通常是 [K, N]
        Kz, Nz = qzeros.shape   # qzeros 通常是 [Kz, Nz]
        # 解压后的零点数组，扩大维度到未压缩后的大小
        unpacked_zeros = np.zeros((Kz, Nz * compress_ratio), dtype=np.uint8)

        # ========== GPTQ 量化格式 ==========
        if quant_mode == "gptq":
            # 解压后的权重，扩展 K 维度
            unpacked_weights = np.zeros((K * compress_ratio, N), dtype=dtype)
            # 额外的 int8 打包权重（用于存储 4bit → 8bit 的紧凑表示）
            pack_int8_weights = np.zeros((K * compress_ratio // 2, N), dtype=np.uint8)
            # GPTQ 的顺序映射（简单按顺序 0,1,2...）
            order_map = [i for i in range(compress_ratio)]
            # 遍历解压后权重的每一行
            for row in range(unpacked_weights.shape[0]):
                # 确定该 row 对应的原始量化块中的第几个值
                i = order_map[row % compress_ratio]
                # 提取 bits 位宽的权重值
                unpacked_weights[row, :] = (qweight[row // compress_ratio, :] >> (bits * i)) & mask
                # 如果是 4bit 量化，则打包为 int8（两个 4bit 合并成 1字节）
                if bits == 4:
                    if row % 2 == 0:
                        # 偶数行先存起来
                        pack_int8_weights[row // 2, :] = unpacked_weights[row, :]
                    else:
                        # 奇数行与之前偶数行合并（高4位+低4位）
                        pack_int8_weights[
                            row //
                            2, :] = unpacked_weights[row, :] << 4 | pack_int8_weights[row // 2, :]
        # ========== AWQ 量化格式 ==========
        elif quant_mode == "awq":
            # AWQ 解压后的权重，扩展 N 维度
            unpacked_weights = np.zeros((K, N * compress_ratio), dtype=dtype)
            # AWQ 的 int8 打包权重（4bit 时同样每两个行合并）
            pack_int8_weights = np.zeros((K // 2, N * compress_ratio), dtype=np.uint8)
            # AWQ 特定顺序映射（和 GPTQ 不同）
            order_map = [0, 4, 1, 5, 2, 6, 3, 7]
            # 遍历所有解压后的列
            for col in range(unpacked_weights.shape[1]):
                # 确定该 col 对应量化块中的位置
                i = order_map[col % compress_ratio]
                # 提取该位置的 bits 位宽权重
                unpacked_weights[:, col] = (qweight[:, col // compress_ratio] >> (bits * i)) & mask
            # 如果是 4bit，则进行 int8 打包（两行合并）
            if bits == 4:
                for row in range(unpacked_weights.shape[0]):
                    if row % 2 == 0:
                        pack_int8_weights[row // 2, :] = unpacked_weights[row, :]
                    else:
                        pack_int8_weights[
                            row //
                            2, :] = unpacked_weights[row, :] << 4 | pack_int8_weights[row // 2, :]
        else:
            # 其他模式暂不支持
            raise NotImplementedError("Not support now")

        # 解压零点信息（zero-points）
        for col in range(unpacked_zeros.shape[1]):
            i = order_map[col % compress_ratio]
            unpacked_zeros[:, col] = (qzeros[:, col // compress_ratio] >> (bits * i)) & mask

        # 如果是 8bit 量化，则打包权重等于解压权重（无额外处理）
        if bits == 8:
            pack_int8_weights = unpacked_weights.astype("uint8")

        # GPTQ 的零点加 1（因为 GPTQ 格式通常零点存储偏移）
        if quant_mode == "gptq":
            return unpacked_weights, pack_int8_weights, unpacked_zeros + 1
        else:
            return unpacked_weights, pack_int8_weights, unpacked_zeros

    # 从模型配置中读取必要参数(隐藏层、注意力头数、维度、激活函数、量化参数等)
    # 处理权重共享和量化特殊逻辑
    def init_config(self):
        # 从模型信息中读取配置对象
        c = self.model_info.config

        # 从 llm_config 中获取对应的属性值
        self.num_layers = getattr(self.llm_config, c.num_hidden_layers)
        # RoPE旋转位置编码的缩放参数
        self.rope_theta = getattr(self.llm_config, c.rope_theta, 10000.0)
        # 注意力头数
        self.num_attention_heads = getattr(self.llm_config, c.num_attention_heads)
        # key/value头数（默认等于 num_attention_heads）
        self.num_key_value_heads = getattr(self.llm_config, c.num_key_value_heads,
                                           self.num_attention_heads)
        # 隐藏层维度
        self.hidden_size = getattr(self.llm_config, c.hidden_size)
        # 词表大小
        self.vocab_size = getattr(self.llm_config, c.vocab_size)
        # FFN 中间层维度
        self.intermediate_size = getattr(self.llm_config, c.intermediate_size)
        # RMSNorm epsilon
        self.rms_norm_eps = getattr(self.llm_config, c.rms_norm_eps)
        # head_dim = hidden_size / num_attention_heads（默认计算值）
        self.head_dim = getattr(self.llm_config, "head_dim",
                                self.hidden_size // self.num_attention_heads)
        # 激活函数，默认使用 SILU
        self.hidden_act = getattr(self.llm_config, c.hidden_act, ActType.SILU)
        # KV缓存的维度和Tile数
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.kv_tile = self.num_attention_heads // self.num_key_value_heads

        # -------- MiniCPM4 额外参数 --------
        self.scale_emb = getattr(self.llm_config, "scale_emb", 1.)# Embedding 缩放系数
        self.scale_depth = getattr(self.llm_config, "scale_depth", 1.)# Depth 缩放系数
        self.dim_model_base = getattr(self.llm_config, "dim_model_base", 1.)# 模型基础维度

        # -------- Embedding & LMHead 权重共享设置 --------
        self.tie_word_embeddings = getattr(self.llm_config, 'tie_word_embeddings', False)# 是否共享embedding和输出层权重
        # 是否合并 lm_head 和 embedding (满足条件：共享权重 + 不使用embedding_disk + 单设备)
        self.do_lmhead_merge = self.tie_word_embeddings and not self.embedding_disk and self.num_device < 2

        # -------- 量化配置 --------
        self.quantization_config = getattr(self.llm_config, c.quantization_config, None)
        if self.quantization_config:
            self.quant_mode = self.quantization_config["quant_method"]# 量化方法（如 gptq/awq）
            self.q_group_size = self.quantization_config["group_size"]# 分组大小
            self.quant_bits = self.quantization_config["bits"]# 量化比特数
            # AWQ 模式只支持 gemm 版本和 4bit
            if self.quant_mode == "awq":
                assert self.quantization_config["version"] == "gemm", (
                    "AWQ only support gemm version for now")
                assert self.quant_bits == 4, ("AWQ only support quant bits == 4 for now")
        # 如果量化分组大小是负数，重置为0（可能表示不分组）
        if self.q_group_size < 0:
            self.q_group_size = 0

    # 将层的名字(单个或多个)转为 MLIR 可用的调试位置信息对象
    def get_loc(self, names, mlir):
        # 将字符串或字符串列表转换为 MLIR 的 Location 对象
        if isinstance(names, str):
            # 单个名字 → 创建 fused location
            return Location.fused([Location.name(names)], context=mlir.ctx)
        elif isinstance(names, list):
            # 多个名字 → 创建 fused location
            return Location.fused([Location.name(n) for n in names], context=mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    # 将嵌入层权重数据转换位bf16或f16格式，并直接写入二进制文件(方便推理时直接加载)
    def gen_embedding_bin(self, embedding_data):
        # 导出 embedding 为二进制文件 ../embedding.bin
        embedding_file = '../embedding.bin'
        # 如果文件已存在，则跳过
        if os.path.exists(embedding_file):
            print(f"{embedding_file} already exists. Skipping export.")
            return
        import ctypes
        # 将 numpy 数据转换为 torch tensor
        weight = torch.from_numpy(embedding_data)
        # 根据量化类型选择数据类型
        if 'bf16' in self.quantize:
            tensor_data = weight.to(torch.bfloat16)
        elif 'f16' in self.quantize:
            tensor_data = weight.to(torch.float16)
        else:
            raise NotImplementedError("Not support now")
        # 获取底层内存指针（直接访问Tensor的存储区）
        data_ptr = tensor_data.untyped_storage().data_ptr()
        # 将内存映射为 ctypes 字节数组（长度 = 元素数 × 2字节）
        buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
        # 写入二进制文件
        with open(embedding_file, 'wb') as f:
            f.write(buffer)

    # 生成嵌入层和语言模型头部的MLIR代码
    def gen_embedding_lmhead_mlir(self):
        """生成嵌入层(embedding)和语言模型头部(lm_head)的MLIR表示"""

        # 打印进度信息
        tqdm.write("generate embedding and lm_head mlir ...")

        # 获取嵌入层权重路径并读取权重数据
        embedding_path = self.model_info.weights[LlmList.EMBEDING] + ".weight"
        embedding_data = self.model.read(embedding_path)

        # 处理嵌入层权重：根据配置决定是保存为二进制文件还是NPZ文件
        if self.embedding_disk:
            # 如果配置要求将嵌入权重保存到磁盘
            self.gen_embedding_bin(embedding_data)
        else:
            # 将嵌入权重保存为NPZ格式文件，供后续MLIR生成使用
            embedding_weights = {embedding_path: embedding_data}
            embedding_npz = "embedding_top_weights.npz"
            np.savez(embedding_npz, **embedding_weights)

        # 获取语言模型头部(lm_head)和归一化层(norm)的路径
        lmhead = self.model_info.weights[LlmList.LMHEAD]
        lmhead_path = lmhead + ".weight"
        norm = self.model_info.weights[LlmList.NORM]
        norm_path = norm + ".weight"

        # 处理语言模型头部权重
        if self.tie_word_embeddings:
            # 如果配置要求词嵌入共享权重，则直接使用嵌入层权重
            lmhead_data = embedding_data
        else:
            # 否则单独读取语言模型头部的权重
            lmhead_data = self.model.read(lmhead_path)

        # 处理语言模型头部权重和归一化层权重
        if not self.do_lmhead_merge:
            # 如果不合并语言模型头部，需要转置权重并包含归一化层权重
            lmhead_data = np.ascontiguousarray(np.transpose(lmhead_data, (1, 0)))
            norm_data = self.model.read(norm_path)
            lmhead_weights = {lmhead_path: lmhead_data, norm_path: norm_data}
        else:
            # 如果合并语言模型头部，只包含语言模型头部权重
            lmhead_weights = {lmhead_path: lmhead_data}

        # 保存语言模型头部权重为NPZ文件
        lmhead_npz = "lm_head_top_weights.npz"
        np.savez(lmhead_npz, **lmhead_weights)

        # 内部函数：生成嵌入层的MLIR表示
        def gen_embedding_by_length(name: str, seq_length: int):
            """为指定序列长度生成嵌入层MLIR

            Args:
                name: 生成的MLIR文件名前缀
                seq_length: 输入序列长度
            """
            # 定义输出形状 [batch_size, sequence_length, hidden_size]
            out_shape = [1, seq_length, self.hidden_size]
            # 创建MLIR导入器对象
            embedding_mlir = MLIRImporter([[1, seq_length]],# 输入形状
                                          [out_shape],# 输出形状
                                          name,# 操作名称
                                          Platform.LLM,# 目标平台
                                          input_types=["INT32"],# 输入类型为整数
                                          weight_file=embedding_npz)# 权重文件

            # 创建输入操作（输入为token ID）
            input_op = embedding_mlir.create_input_op(self.get_loc("input_ids", embedding_mlir), 0)
            # 创建权重操作（嵌入矩阵 [vocab_size, hidden_size]）
            weight_op = embedding_mlir.create_weight_op(embedding_path,
                                                        [self.vocab_size, self.hidden_size])

            # 创建Gather操作：根据输入token ID查找嵌入向量
            new_op = top.GatherOp(embedding_mlir.get_tensor_type(out_shape),# 输出张量类型
                                  weight_op,# 权重操作
                                  input_op,# 输入操作
                                  axis=0,# 在词汇表维度（第0维）进行查找
                                  loc=self.get_loc(name, embedding_mlir),
                                  ip=embedding_mlir.insert_point).output

            # 特定模型处理：GEMMA3模型需要对嵌入向量进行缩放
            if self.llm_type in [LlmType.GEMMA3]:
                new_op = top.MulConstOp(embedding_mlir.get_tensor_type(out_shape),
                                        new_op,
                                        const_val=self.hidden_size**0.5,# 缩放因子为hidden_size的平方根
                                        loc=self.get_loc(name + ".scale", embedding_mlir),
                                        ip=embedding_mlir.insert_point).output
            # 特定模型处理：MINICPM4模型需要对嵌入向量进行缩放
            if self.llm_type == LlmType.MINICPM4:
                new_op = top.MulConstOp(embedding_mlir.get_tensor_type(out_shape),
                                        new_op,
                                        const_val=self.scale_emb,# 使用预定义的缩放因子
                                        loc=self.get_loc(name + ".scale", embedding_mlir),
                                        ip=embedding_mlir.insert_point).output
            # 创建返回操作，结束图构建
            embedding_mlir.create_return_op([new_op])
            # 打印MLIR模块并写入文件
            mlir_txt = embedding_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # 内部函数：生成语言模型头部的MLIR表示
        def gen_lm_head():
            """生成语言模型头部的MLIR表示"""

            # 根据配置确定输出形状
            out_shape = [[1, self.vocab_size]]
            if self.lmhead_with_topk:
                # 如果使用TopK操作，输出形状为[1, 1]（只输出一个token）
                out_shape = [[1, 1]]
            # 创建MLIR导入器对象
            lmhead_mlir = MLIRImporter([[1, self.hidden_size]],# 输入形状 [batch_size, hidden_size]
                                       out_shape,# 输出形状
                                       "lm_head",# 操作名称
                                       Platform.LLM,# 目标平台
                                       weight_file=lmhead_npz)# 权重文件

            # 创建输入操作（输入为隐藏状态）
            input_op = lmhead_mlir.create_input_op(self.get_loc("hidden_states", lmhead_mlir), 0)
            # 处理未合并的语言模型头部
            if not self.do_lmhead_merge:
                # 创建归一化层权重操作
                weight_op = lmhead_mlir.create_weight_op(norm_path, [1, self.hidden_size])
                # 应用RMS归一化
                input_op = self.rms_norm(lmhead_mlir, input_op, norm)
                # MINICPM4模型需要额外的缩放操作
                if self.llm_type == LlmType.MINICPM4:
                    input_op = top.MulConstOp(lmhead_mlir.get_tensor_type([1, self.hidden_size]),
                                              input_op,
                                              const_val=self.dim_model_base / self.hidden_size,# 缩放因子
                                              loc=self.get_loc(lmhead + ".scale", lmhead_mlir),
                                              ip=lmhead_mlir.insert_point).output
                # 定义线性层权重形状 [hidden_size, vocab_size]
                w_shape = [self.hidden_size, self.vocab_size]
                # 创建线性层（全连接层）
                lmhead_op = self.linear(lmhead_mlir,# MLIR导入器
                                        lmhead,# 操作名称
                                        input_op,# 输入操作
                                        w_shape,# 权重形状
                                        [1, self.vocab_size])# 输出形状
            else:
                # 处理合并的语言模型头部
                w_shape = [self.vocab_size, self.hidden_size]
                # 创建权重操作
                weight_op = lmhead_mlir.create_weight_op(lmhead + ".weight", w_shape)
                # 创建矩阵乘法操作（注意：right_transpose=True表示对权重进行转置）
                lmhead_op = top.MatMulOp(lmhead_mlir.get_tensor_type([self.vocab_size, 1]),
                                         weight_op,
                                         input_op,
                                         lmhead_mlir.none_op,
                                         do_relu=False,
                                         right_transpose=True,# 对权重进行转置：W^T
                                         loc=self.get_loc(lmhead, lmhead_mlir),
                                         ip=lmhead_mlir.insert_point).output
                # 重塑输出形状为 [1, vocab_size]
                lmhead_op = top.ReshapeOp(lmhead_mlir.get_tensor_type([1, self.vocab_size]),
                                          lmhead_op,
                                          loc=self.get_loc(lmhead + ".reshape", lmhead_mlir),
                                          ip=lmhead_mlir.insert_point).output
            # 处理TopK操作（如果启用）
            if self.lmhead_with_topk:
                # 创建TopK操作，获取概率最大的token
                topk_op = top.TopKOp(*lmhead_mlir.get_tensor_type([[1, 1], [1, 1]]),# 输出类型（值和索引）
                                     lmhead_op,# 输入操作
                                     axis=1,# 在词汇表维度执行TopK
                                     K=1,# 只取概率最大的一个token
                                     loc=self.get_loc(["token_value", "token_id"], lmhead_mlir),
                                     ip=lmhead_mlir.insert_point)
                # 返回token索引（token_id）
                lmhead_mlir.create_return_op([topk_op.indices])
            else:
                # 直接返回语言模型头部的输出（整个词汇表的logits）
                lmhead_mlir.create_return_op([lmhead_op])
            # 打印MLIR模块并写入文件
            mlir_txt = lmhead_mlir.print_module()
            with open("lm_head.mlir", "w") as f:
                f.write(mlir_txt)

        # 生成嵌入层MLIR（如果权重没有保存到磁盘）
        if not self.embedding_disk:
            # 为最大输入长度生成嵌入层
            gen_embedding_by_length("embedding", self.max_input_length)
            # 为缓存生成嵌入层（序列长度为1）
            gen_embedding_by_length("embedding_cache", 1)
        # 生成语言模型头部MLIR
        gen_lm_head()

    # 生成Greedy和Sample两种解码头部的MLIR表示
    def gen_sample_head_mlir(self, max_top_k=50, min_tokens_to_keep=5):
        # 打印当前操作信息（tqdm.write可在进度条中显示）
        tqdm.write("generate greedy head and sample head mlir ...")
        ############## Greedy Head（贪心选择） ##############

        # 创建MLIR导入器对象（输入：[batch_size, vocab_size]，输出：[batch_size, 1]）
        greedy_head_mlir = MLIRImporter([[1, self.vocab_size]], [[1, 1]],
                                        "greedy_head",# 操作名称
                                        Platform.LLM,# LLM平台
                                        weight_file=None)# 无额外权重

        # 创建输入操作符（从logits中获取概率分布）
        input_op = greedy_head_mlir.create_input_op(self.get_loc("m_logits", greedy_head_mlir), 0)

        # 构建TopK操作（K=1即选择最高概率的token）
        topk_op = top.TopKOp(*greedy_head_mlir.get_tensor_type([[1, 1], [1, 1]]),# 输出类型（值和索引）
                             input_op,# 输入张量
                             axis=1,# 沿vocab维度处理
                             K=1,# 只取Top1
                             loc=self.get_loc(["token_value", "token_id"], greedy_head_mlir),# 位置信息
                             ip=greedy_head_mlir.insert_point)# 插入点

        # 创建返回操作（只返回token索引）
        greedy_head_mlir.create_return_op([topk_op.indices])

        # 将MLIR模块写入文件
        mlir_txt = greedy_head_mlir.print_module()
        with open(f"greedy_head.mlir", "w") as f:
            f.write(mlir_txt)

        ############## Sample Head（采样选择） ##############

        # 准备采样所需的常量权重（用于确保保留的最小token数量）
        constant0 = []  # 索引位置
        constant1 = []  # 赋值标志
        for i in range(min_tokens_to_keep):
            constant0.append([0, i])# 第0批次的第i个位置
            constant1.append(1)# 赋值为1

        # 构建采样头的权重字典
        sample_head_weights = {}
        sample_head_weights["Constant0"] = np.array([1.]).astype(np.float32)# 标量常数1
        sample_head_weights["Constant1"] = np.array([constant0]).astype(np.float32)# 最小token保留位置
        sample_head_weights["Constant2"] = np.array([constant1]).astype(np.float32)# 保留标记
        np.savez("sample_head_top_weights.npz", **sample_head_weights)# 保存权重

        # 创建采样头MLIR导入器（更复杂的输入输出结构）
        sample_head_mlir = MLIRImporter(
            [[1, self.vocab_size],# logits
             [1, self.seq_length],# input_ids
             [1], [1], [1], [1]],# 惩罚/temperature/top_k/top_p参数

            [[1, max_top_k], [1, max_top_k]],# 采样值+索引输出
            "sample_head",# 操作名称
            Platform.LLM,# LLM平台
            input_types=['F32', 'INT32', 'F32', 'F32', 'INT32', 'F32'],# 各输入类型
            weight_file="sample_head_top_weights.npz")# 预定义权重文件
        ip = sample_head_mlir.insert_point# 获取当前插入点
        # 定义类型和位置简写函数
        def T(shape: list):
            return sample_head_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, sample_head_mlir)
        ###### 构建采样计算图 ######

        # 1. 输入操作定义
        kwargs = {}
        kwargs['shape_tensor'] = [max_top_k]
        in0_op = sample_head_mlir.create_input_op(L("m_logits"), 0)# logits
        in1_op = sample_head_mlir.create_input_op(L("input_ids"), 1)# input_ids
        in2_op = sample_head_mlir.create_input_op(L("penalty"), 2)# penalty（重复惩罚）
        in3_op = sample_head_mlir.create_input_op(L("temperature"), 3)# 温度参数
        in4_op = sample_head_mlir.create_input_op(L("top_k"), 4, kwargs)# top_k参数
        in5_op = sample_head_mlir.create_input_op(L("top_p"), 5)# top_p参数

        # 2. 惩罚调整（降低重复token概率）
        # 2.1 获取当前输入对应的logits
        gather_op = top.GatherElementsOp(T([1, self.seq_length]),
                                         in0_op,
                                         in1_op,
                                         axis=1,
                                         loc=L("GatherElements"),
                                         ip=ip).output

        # 2.2 识别低概率值（<0）进行调整
        cmpconst_op = top.CompareConstOp(T([1, self.seq_length]),
                                         gather_op,
                                         mode=StringAttr.get("Less"),# 小于0的判断
                                         const_val=0.,
                                         inversed=False,
                                         loc=L("CompareConst"),
                                         ip=ip).output

        # 2.3 应用惩罚：对负值区域应用惩罚因子（放大差异）
        mul_op = top.MulOp(T([1, self.seq_length]), [gather_op, in2_op], loc=L("Mul"), ip=ip).output
        # 对正值区域应用抑制因子（缩小差异）
        div0_op = top.DivOp(T([1, self.seq_length]), [gather_op, in2_op], loc=L("Div0"),
                            ip=ip).output
        # 合并调整后的结果
        where0_op = top.WhereOp(T([1, self.seq_length]),
                                cmpconst_op,# 条件（负值区域）
                                mul_op,# 负值处理结果
                                div0_op,# 正值处理结果
                                loc=L("Where0"),
                                ip=ip).output
        # 2.4 将调整后的值填充回完整logits矩阵
        scatter_op = top.ScatterElementsOp(T([1, self.vocab_size]),
                                           in0_op,# 原始logits
                                           in1_op,# input_ids位置
                                           where0_op,# 调整后的值
                                           axis=1,# 沿vocab维度
                                           loc=L("ScatterElements"),
                                           ip=ip).output
        # 3. 采样核心流程
        # 3.1 执行TopK（基于max_top_k参数取概率最高的候选）
        topk_op = top.TopKOp(*T([[1, max_top_k], [1, max_top_k]]),
                             scatter_op,# 调整后的logits
                             kT=in4_op,# top_k参数（动态值）
                             axis=1,
                             K=max_top_k,# 最大候选数
                             loc=L(["token_value", "token_idx"]),
                             ip=ip)
        # 3.2 温度调节（控制分布的尖锐程度）
        div1_op = top.DivOp(T([1, max_top_k]),
                            [topk_op.values, in3_op],# top_k值/温度
                             loc=L("Div1"),
                            ip=ip).output

        # 3.3 创建临时概率分布（带温度调节）
        softmax0_op = top.SoftmaxOp(T([1, max_top_k]), div1_op, axis=1, loc=L("Softmax0"),
                                    ip=ip).output
        # 3.4 累积概率计算（用于top_p/nucleus采样）
        weight0_op = sample_head_mlir.create_weight_op("Constant0", [1])# 权重1（步长）

        cumsum_op = top.CumSumOp(T([1, max_top_k]),
                                 softmax0_op,# softmax结果
                                 weight0_op,# 累加起始值
                                 axis=1,# 沿top_k维度
                                 loc=L("CumSum"),
                                 ip=ip).output
        # 3.5 top_p过滤（保留累积概率小于p的候选）
        compare_op = top.CompareOp(T([1, max_top_k]),
                                   cumsum_op,# 累积概率
                                   in5_op,# top_p阈值
                                   mode=StringAttr.get("Less"),# 小于阈值
                                   loc=L("Compare"),
                                   ip=ip).output

        # 3.6 确保保留min_tokens_to_keep个token（即使不满足top_p）
        weight1_op = sample_head_mlir.create_weight_op("Constant1", [1, min_tokens_to_keep, 2])# 位置
        weight2_op = sample_head_mlir.create_weight_op("Constant2", [1, min_tokens_to_keep])# 值

        # 将前N个token强制标记为有效
        scatternd_op = top.ScatterNDOp(T([1, max_top_k]),
                                       compare_op,# 原有有效性标记
                                       weight1_op,# 要强制设置的位置
                                       weight2_op,# 要设置的值（1=有效）
                                       reduction=0,# 覆盖模式
                                       loc=L("ScatterND"),
                                       ip=ip).output

        # 3.7 过滤掉无效候选（替换为极小值-1000）
        where1_op = top.WhereOp(T([1, max_top_k]),
                                scatternd_op,# 最终的候选有效性标记
                                div1_op,# 原始分数（有效候选保留）
                                sample_head_mlir.none_op,# None表示使用常量替代
                                y_is_const=True,# 使用常量替代
                                y_const_val=-1000.,# 替代值（极小值）
                                loc=L("Where1"),
                                ip=ip).output

        # 3.8 创建最终采样概率分布（过滤后的softmax）
        softmax1_op = top.SoftmaxOp(T([1, max_top_k]),
                                    where1_op,# 过滤后的候选分数
                                    axis=1, loc=L("Softmax1"),
                                    ip=ip).output
        # 创建返回操作（返回采样概率和候选索引）
        sample_head_mlir.create_return_op([softmax1_op, topk_op.indices])
        # 将MLIR模块写入文件
        mlir_txt = sample_head_mlir.print_module()

        with open(f"sample_head.mlir", "w") as f:
            f.write(mlir_txt)

    def repeat_kv(self, mlir_gen, kv_op, len: int, prefix: str):
        # 该函数用于扩展（重复）Key/Value缓存的维度，使其从 (num_key_value_heads)
        # 转换成 (num_attention_heads)，因为有些模型使用了头分组（key/value头数 < 注意力头数）

        # 1. 先在原始KV张量的维度上插入一个维度（axes=[3]表示在第3维插入）
        #   原始张量形状: [1, len, num_key_value_heads, head_dim]
        #   新形状: [1, len, num_key_value_heads, 1, head_dim]
        unsqueeze = top.UnsqueezeOp(mlir_gen.get_tensor_type(
            [1, len, self.num_key_value_heads, 1, self.head_dim]),
                                    kv_op,
                                    loc=self.get_loc(prefix + ".unsqueeze", mlir_gen),
                                    ip=mlir_gen.insert_point,
                                    axes=[3]).output

        # 2. 使用Tile操作在第3维(新插入的维度)重复kv_tile次（kv_tile = num_attention_heads / num_key_value_heads）
        #   结果形状: [1, len, num_key_value_heads, kv_tile, head_dim]
        tile = top.TileOp(mlir_gen.get_tensor_type(
            [1, len, self.num_key_value_heads, self.kv_tile, self.head_dim]),
                          unsqueeze,
                          tile=[1, 1, 1, self.kv_tile, 1],
                          loc=self.get_loc(prefix + ".tile", mlir_gen),
                          ip=mlir_gen.insert_point).output

        # 3. 重塑(Reshape)张量，将(num_key_value_heads * kv_tile) 合并成 num_attention_heads
        #   最终形状: [1, len, num_attention_heads, head_dim]
        rs = top.ReshapeOp(mlir_gen.get_tensor_type(
            [1, len, self.num_attention_heads, self.head_dim]),
                           tile,
                           loc=self.get_loc(prefix + ".tile.reshape", mlir_gen),
                           ip=mlir_gen.insert_point).output
        return rs

    def linear(self,
               mlir_gen,
               proj: str,
               input_op,
               weight_shape: list,
               out_shape: list,
               force_bias: bool = False):

        # 通用的线性层（MatMul + Bias）
        # 支持普通浮点权重和量化权重两种情况

        # 创建权重Op (proj.weight)
        weight_op = mlir_gen.create_weight_op(proj + ".weight", weight_shape)

        # 判断是否存在bias或强制添加bias
        if self.model.is_exist(proj + ".bias") or force_bias:
            # bias的形状: 广播兼容 [1,1,..., out_features]
            bias_shape = [1] * (len(out_shape) - 1) + [out_shape[-1]]
            bias_op = mlir_gen.create_weight_op(proj + ".bias", bias_shape)
        else:
            # 无bias时传入none_op
            bias_op = mlir_gen.none_op

        # ========== 量化权重分支 ==========
        if self.quant_mode and self.model.is_exist(proj + ".qweight"):
            # 量化权重 (proj.qweight)，存储为uint8
            # shape: [out_features/(8/bits), in_features]
            qweight_op = mlir_gen.create_weight_op(
                proj + ".qweight", [weight_shape[1], weight_shape[0] // (8 // self.quant_bits)],
                'UINT8')
            # 量化scale和零点
            scale_shape = [weight_shape[1], weight_shape[0] //
                           self.q_group_size] if self.q_group_size > 0 else [weight_shape[1], 1]

            scale_op = mlir_gen.create_weight_op(proj + ".scales", scale_shape)
            zp_op = mlir_gen.create_weight_op(proj + ".qzeros", scale_shape, 'UINT8')
            # 调用支持量化的矩阵乘法A16MatMulOp
            return top.A16MatMulOp(mlir_gen.get_tensor_type(out_shape),
                                   input_op,
                                   qweight_op,
                                   scale_op,
                                   zp_op,
                                   bias_op,
                                   right_transpose=True,# 表示权重矩阵转置
                                   q_group_size=self.q_group_size,
                                   weight_bits=self.quant_bits,
                                   loc=self.get_loc(proj, mlir_gen),
                                   ip=mlir_gen.insert_point).output
        # ========== 普通权重分支 ==========
        weight_op = mlir_gen.create_weight_op(proj + ".weight", weight_shape)
        # 使用普通MatMulOp
        return top.MatMulOp(mlir_gen.get_tensor_type(out_shape),
                            input_op,
                            weight_op,
                            bias_op,
                            do_relu=False,
                            loc=self.get_loc(proj, mlir_gen),
                            ip=mlir_gen.insert_point).output

    def rotary_pos(self, mlir_gen, in_op, cos_op, sin_op, out_name: str):
        """
        实现旋转位置编码（Rotary Position Embedding）的核心操作

        参数:
            mlir_gen: MLIR生成器对象
            in_op: 输入张量操作（通常是query或key向量）
            cos_op: 余弦位置编码张量操作
            sin_op: 正弦位置编码张量操作
            out_name: 输出操作的名称前缀
        """
        # 获取输入张量的形状 [batch_size, seq_len, num_heads, head_dim]
        in_shape = in_op.type.shape
        prefix = f"{out_name}.rotary_pos"

        # 计算半头维度（将头维度分为两部分）
        half_shape = list(in_shape)
        half_shape[-1] = half_shape[-1] // 2# 将最后一个维度(head_dim)分成两半

        # 第一部分：输入与余弦位置编码相乘
        mul_q_proj = top.MulOp(mlir_gen.get_tensor_type(in_shape),# 输出类型
                               [in_op, cos_op],# 输入操作
                               loc=self.get_loc(prefix + ".mul0", mlir_gen),
                               ip=mlir_gen.insert_point).output

        # 第二部分：切分输入张量为两部分

        # 提取输入的前半部分 [..., 0:head_dim/2]
        half_q0 = top.SliceOp(mlir_gen.get_tensor_type(half_shape),# 输出类型
                              in_op,# 输入操作
                              mlir_gen.none_op,# 未使用的参数
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              offset=[0, 0, 0, 0],# 起始索引（所有维度的起始位置）
                              steps=[1, 1, 1, 1],# 步长（不跳过任何元素）
                              ends=half_shape,# 结束索引（前半部分的结束位置）
                              axes=[],# 在所有维度上切片
                              loc=self.get_loc(prefix + ".slice1", mlir_gen),
                              ip=mlir_gen.insert_point).output
        # 提取输入的后半部分 [..., head_dim/2:head_dim]
        half_q1 = top.SliceOp(mlir_gen.get_tensor_type(half_shape),# 输出类型
                              in_op,# 输入操作
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              offset=[0, 0, 0, half_shape[-1]],# 起始索引（最后一个维度从中间开始）
                              steps=[1, 1, 1, 1],
                              ends=in_shape,# 结束索引（到原始输入的末尾）
                              axes=[],
                              loc=self.get_loc(prefix + ".slice2", mlir_gen),
                              ip=mlir_gen.insert_point).output

        # 第三部分：创建旋转后的向量

        # 对后半部分取负（旋转操作的关键步骤）
        neg_half_q1 = top.MulConstOp(mlir_gen.get_tensor_type(half_shape),# 输出类型
                                     half_q1,# 输入操作
                                     const_val=-1.0,# 乘以-1（相当于取负）
                                     loc=self.get_loc(prefix + ".neg3", mlir_gen),
                                     ip=mlir_gen.insert_point).output
        # 将负后半部分和前半部分拼接起来 [ -后半部分, 前半部分 ]
        new_q = top.ConcatOp(mlir_gen.get_tensor_type(in_shape),# 输出类型（与原始输入相同）
                             [neg_half_q1, half_q0],# 要拼接的张量
                             axis=3,# 在最后一个维度（head_dim）拼接
                             loc=self.get_loc(prefix + ".concat4", mlir_gen),
                             ip=mlir_gen.insert_point).output
        # 第四部分：应用正弦位置编码
        new_q = top.MulOp(mlir_gen.get_tensor_type(in_shape),# 输出类型
                          [new_q, sin_op],# 输入操作（旋转后的向量和正弦位置编码）
                          loc=self.get_loc(prefix + ".mul5", mlir_gen),
                          ip=mlir_gen.insert_point).output
        # 第五部分：组合最终结果
        new_q = top.AddOp(mlir_gen.get_tensor_type(in_shape),# 输出类型
                          [mul_q_proj, new_q],# 输入操作（余弦部分 + 正弦部分）
                          loc=self.get_loc(out_name, mlir_gen),# 最终操作的名称
                          ip=mlir_gen.insert_point).output
        return new_q

    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        """
        将旋转位置编码应用到query和key向量上

        参数:
            mlir_gen: MLIR生成器对象
            pos_op: 位置索引张量操作
            q_op: query向量操作
            k_op: key向量操作
            rotary_cos: 余弦位置编码的权重路径
            rotary_sin: 正弦位置编码的权重路径
        """
        # 获取位置张量的维度
        dim = pos_op.type.shape[-1]
        # 创建余弦位置编码操作
        weight_op = mlir_gen.create_weight_op(rotary_cos + ".weight",# 权重路径
                                              [self.seq_length, 1, self.head_dim])# 权重形状 [序列长度, 1, 头维度]

        cos_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),# 输出类型
                              weight_op,# 权重操作
                              pos_op,# 位置索引操作
                              axis=0,# 在序列长度维度上查找
                              loc=self.get_loc(rotary_cos, mlir_gen),# 操作位置
                              ip=mlir_gen.insert_point).output# 插入点
        # 创建正弦位置编码操作（与余弦类似）
        weight_op = mlir_gen.create_weight_op(rotary_sin + ".weight",# 权重路径
                                              [self.seq_length, 1, self.head_dim])# 权重形状

        sin_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),# 权重形状
                              weight_op,# 权重操作
                              pos_op,# 位置索引操作
                              axis=0,# 在序列长度维度上查找
                              loc=self.get_loc(rotary_sin, mlir_gen),# 操作位置
                              ip=mlir_gen.insert_point).output# 插入点

        # ===== 应用到query向量 ========
        q_op = self.rotary_pos(mlir_gen, q_op, cos_op, sin_op, "q_proj")

        # ===== 应用到key向量 ========
        k_op = self.rotary_pos(mlir_gen, k_op, cos_op, sin_op, "k_cache")

        return q_op, k_op

    def set_linear_weight(self, path: str, weight_dict: dict):
        """
        设置线性层权重，支持量化和非量化两种情况

        参数:
            path: 权重路径前缀
            weight_dict: 用于存储权重的字典
        """
        # 检查是否为量化模式
        is_quant = self.quant_mode is not None and self.model.is_exist(path + ".qweight")
        if not is_quant:
            # 非量化模式处理
            weight_path = path + ".weight"
            bias_path = path + ".bias"
            if self.model.is_exist(weight_path):
                # 读取权重数据并转置（PyTorch线性层权重是[out_features, in_features]）
                data = self.model.read(weight_path)
                weight_dict[weight_path] = np.ascontiguousarray(np.transpose(data, (1, 0)))
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        else:
            # 量化模式处理
            qweight_path = path + ".qweight"  # 量化权重
            scale_path = path + ".scales"     # 缩放因子
            zp_path = path + ".qzeros"        # 零点（zero point）
            bias_path = path + ".bias"        # 偏置
            if self.model.is_exist(qweight_path):
                # 读取量化权重数据
                qweight_data = self.model.read(qweight_path)
                scale_data = self.model.read(scale_path)
                zp_data = self.model.read(zp_path)
                # 解包量化权重
                _, pack_int8_weights, unpacked_zeros = self.unpack_weights(
                    qweight_data, zp_data, self.quant_bits, self.quant_mode)

                # 存储转置后的量化权重
                weight_dict[qweight_path] = np.ascontiguousarray(
                    np.transpose(pack_int8_weights, (1, 0)))
                # 存储转置后的缩放因子
                weight_dict[scale_path] = np.ascontiguousarray(np.transpose(scale_data, (1, 0)))
                # 存储解包后的零点
                weight_dict[zp_path] = np.ascontiguousarray(np.transpose(unpacked_zeros, (1, 0)))
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        # 处理偏置项（量化和非量化模式都需要）
        if self.model.is_exist(bias_path):
            weight_dict[bias_path] = self.model.read(bias_path)

    def set_common_weight(self, path: str, weight_dict: dict, type=WeightType.NORMAL):
        # 该函数用于从模型文件中读取给定路径对应的权重，并填充到 weight_dict 字典中
        # 支持普通权重和 RMSNorm 特殊处理（针对 GEMMA3 模型）

        # 拼接可能存在的权重和偏置路径
        weight_path = path + ".weight"
        bias_path = path + ".bias"

        # 检查模型文件中是否存在对应键
        has_weight = self.model.is_exist(weight_path)# 是否有权重文件
        has_bias = self.model.is_exist(bias_path)# 是否有偏置文件
        has_path = self.model.is_exist(path)# 某些模型权重可能直接存在于 path

        # 如果三者都不存在，则抛出异常
        if not has_weight and not has_bias and not has_path:
            raise RuntimeError("Can't find key: {}".format(path))

        # 如果存在 weight
        if has_weight:
            data = self.model.read(weight_path)

            # 如果是 RMS_NORM 类型，并且模型是 GEMMA3，则对权重加 1.0
            # （因为 GEMMA3 的 RMSNorm 权重存储规则不同）
            if type == WeightType.RMS_NORM and self.llm_type in [LlmType.GEMMA3]:
                data = data + 1.0  # GEMMA3 RMSNorm weight is not same as others
            # 将读取的权重存入 weight_dict
            weight_dict[weight_path] = data
        # 如果存在 bias
        if has_bias:
            weight_dict[bias_path] = self.model.read(bias_path)
        # 如果存在 path 本身对应的权重（可能是特殊存储格式）
        if has_path:
            weight_dict[path] = self.model.read(path)

    # 创建Transformer Block的MLIR表示（三种模式）
    def gen_block_mlir(self, idx: int):
        # 生成单个Transformer Block的MLIR表示
        # idx: 当前Block的索引
        tqdm.write(f"generate block_{idx} mlir ...")
        # torch path
        # ====== 获取该Block相关的权重路径前缀 ======
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]# 输入层归一化权重
        q_proj = TOP_PATH + self.model_info.weights[LlmList.Q_PROJ]# Q投影层权重
        q_norm = TOP_PATH + self.model_info.weights[LlmList.Q_NORM]# Q归一化权重（某些模型用）
        k_proj = TOP_PATH + self.model_info.weights[LlmList.K_PROJ]# K投影层权重
        k_norm = TOP_PATH + self.model_info.weights[LlmList.K_NORM]# K归一化权重（某些模型用）
        v_proj = TOP_PATH + self.model_info.weights[LlmList.V_PROJ]# V投影层权重
        o_proj = TOP_PATH + self.model_info.weights[LlmList.O_PROJ]# 输出投影层权重
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]# Attention后归一化
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]# MLP门控权重
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]# MLP上投影权重
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]# MLP下投影权重
        # GEMMA3模型有额外的前置/后置MLP归一化层
        if self.llm_type in [LlmType.GEMMA3]:
            pre_mlp_ln = TOP_PATH + self.model_info.weights[LlmList.PRE_MLP_LN]
            post_mlp_ln = TOP_PATH + self.model_info.weights[LlmList.POST_MLP_LN]
        # 整体模型最后的Norm（用于最后一层）
        norm = self.model_info.weights[LlmList.NORM]
        # 如果是最后一层且需要融合lm_head，则做额外归一化
        do_norm = self.do_lmhead_merge and idx == self.num_layers - 1
        # RoPE位置编码权重
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"

        # ====== 保存权重到npz文件 ======
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }

        # 依次将本block需要的权重加载进字典
        self.set_common_weight(input_ln, weight_dict, WeightType.RMS_NORM)
        self.set_linear_weight(q_proj, weight_dict)
        self.set_linear_weight(k_proj, weight_dict)
        self.set_linear_weight(v_proj, weight_dict)
        self.set_linear_weight(o_proj, weight_dict)
        if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
            self.set_common_weight(q_norm, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(k_norm, weight_dict, WeightType.RMS_NORM)
        if self.llm_type in [LlmType.GEMMA3]:
            self.set_common_weight(pre_mlp_ln, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(post_mlp_ln, weight_dict, WeightType.RMS_NORM)
        self.set_common_weight(post_attn_ln, weight_dict, WeightType.RMS_NORM)
        self.set_linear_weight(mlp_gate, weight_dict)
        self.set_linear_weight(mlp_up, weight_dict)
        self.set_linear_weight(mlp_down, weight_dict)
        if do_norm:
            self.set_common_weight(norm, weight_dict, WeightType.RMS_NORM)

        # 保存到npz文件（方便后续MLIR加载）
        np.savez(weight_file, **weight_dict)

        # ========== 定义MLP子图生成函数 ==========
        def gen_mlp(mlir_gen, input_shape, in_op):

            # mlir_gen: 用于生成MLIR算子的对象
            # input_shape: 输入张量形状
            # in_op: 上一层的输出Op
            ip = mlir_gen.insert_point# 获取当前插入点
            len = input_shape[1]
            new_op = in_op

            # 1. MLP前的归一化处理
            if self.llm_type in [LlmType.GEMMA3]:
                # GEMMA3: 使用前置MLP归一化
                new_op = self.rms_norm(mlir_gen, in_op, pre_mlp_ln)
            else:
                # 其他模型: 使用Attention后的归一化
                new_op = self.rms_norm(mlir_gen, in_op, post_attn_ln)

            # 2. MLP Gate分支
            gate_op = self.linear(mlir_gen, mlp_gate, new_op,
                                  [self.hidden_size, self.intermediate_size],
                                  [1, len, self.intermediate_size])
            act_op = self.activate(mlir_gen, gate_op, self.hidden_act, mlp_gate)

            # 3. MLP Up分支
            up_op = self.linear(mlir_gen, mlp_up, new_op,
                                [self.hidden_size, self.intermediate_size],
                                [1, len, self.intermediate_size])

            # 4. Gate * Up 作为MLP中间输出
            new_op = top.MulOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                               [act_op, up_op],
                               loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                               ip=ip).output

            # 5. Down投影回原始hidden维度
            down_op = self.linear(mlir_gen, mlp_down, new_op,
                                  [self.intermediate_size, self.hidden_size], input_shape)

            # 6. 对GEMMA3模型的MLP输出再做一次归一化
            if self.llm_type in [LlmType.GEMMA3]:
                down_op = self.rms_norm(mlir_gen, down_op, post_mlp_ln)

            # 7. MiniCPM4特殊深度缩放处理
            if self.llm_type == LlmType.MINICPM4:
                down_op = top.MulConstOp(mlir_gen.get_tensor_type(input_shape),
                                         down_op,
                                         const_val=self.scale_depth / np.sqrt(self.num_layers),
                                         loc=self.get_loc(mlp_down + ".scale", mlir_gen),
                                         ip=ip).output

            # 8. 残差连接 (Add)
            last_name = "output_states"
            new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output

            # 9. 如果是最后一层且要融合lm_head，做最终归一化
            if do_norm:
                new_op = self.rms_norm(mlir_gen, new_op, norm, last_name)

            return new_op

        ########## 预填充模式（完整序列处理） ##########
        def gen_block():
            name = f"block_{idx}"# Block名称
            input_len = self.max_input_length# 最大输入长度
            input_shape = [1, input_len, self.hidden_size]# 输入状态形状 [batch=1, seq_len, hidden_size]
            id_shape = list(self.position_shape)# 位置ID形状 [1, seq_len]
            mask_shape = [1, 1, input_len, input_len]# 注意力掩码形状 [batch=1, head=1, seq_len, seq_len]

            # Q/K/V投影后的形状
            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]# Q形状
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]# K/V形状

            # 创建MLIR导入器
            block_mlir = MLIRImporter([input_shape, id_shape, mask_shape],# 输入：状态/位置ID/掩码
                                      [input_shape, kv_shape, kv_shape],# 输出：新状态/K/V缓存
                                      name,
                                      Platform.LLM, ["F32", "INT32", "F32"],# 输入数据类型
                                      weight_file=weight_file)
            # 类型和位置简写函数
            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point# 当前插入点

            # 定义输入操作符
            in0_op = block_mlir.create_input_op(L("input_states"), 0)# 输入状态
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)# 位置ID
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)# 注意力掩码
            return_ops = []# 返回操作列表

            # ======== 输入归一化 ========
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # ======== Q/K/V投影 ========
            q_dim = self.num_attention_heads * self.head_dim        # Q总维度
            # Q投影
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, q_dim],# 权重形状
                               [1, input_len, q_dim])# 输出形状
            # K投影
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])

            # V投影
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])
            # 重塑形状以适应多头注意力
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
            # QWEN3/GEMMA3的特殊处理：Q/K归一化
            if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)

            # ======== 旋转位置编码 ========
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)# K缓存作为返回
            return_ops.append(v_op)# V缓存作为返回


            # ======== 注意力机制 ========
            fa_op = top.FAttentionOp(T([1, input_len, q_dim]),# 输出形状
                                     q_op,# 输入Q/K/V/掩码
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,# 无额外输入
                                     scale=self.head_dim**-0.5,# 缩放因子
                                     batch=1,# 批大小
                                     q_head=self.num_attention_heads,# Q头数
                                     kv_head=self.num_key_value_heads,# KV头数
                                     dim=self.head_dim,# 头维度
                                     mq=input_len,# Q序列长度
                                     mk=input_len,# K序列长度
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            # ======== 输出投影 ========
            o_op = self.linear(block_mlir, o_proj, fa_op, [q_dim, self.hidden_size], # 权重形状
                               input_shape)# 输出形状

            # GEMMA3：注意力后归一化
            if self.llm_type == LlmType.GEMMA3:
                o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)

            # MiniCPM4：特殊缩放
            if self.llm_type == LlmType.MINICPM4:
                o_op = top.MulConstOp(T(input_shape),
                                      o_op,
                                      const_val=self.scale_depth / np.sqrt(self.num_layers),# 缩放因子
                                      loc=L(o_proj + ".scale"),
                                      ip=ip).output

            # ======== 残差连接 ========
            o_op = top.AddOp(T(input_shape), [in0_op, o_op],# 原始输入 + 注意力输出
                              loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            # ======== MLP处理 ========
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            # 创建返回操作：新状态 + K/V缓存
            block_mlir.create_return_op([new_op] + return_ops)
            # 输出MLIR文件
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        ########## 解码模式（增量生成） ##########
        def gen_block_cache():
            name = f"block_cache_{idx}"# 名称标识
            input_shape = [1, 1, self.hidden_size]# 输入形状：每次1个token
            id_shape = list(self.position_shape)# 位置ID形状（仅当前token）
            id_shape[-1] = 1
            mask_shape = [1, 1, 1, self.seq_length + 1]# 注意力掩码（历史+当前）
            history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]# 历史KV形状

            # Q/K/V投影后形状
            q_shape = [1, 1, self.num_attention_heads, self.head_dim]# 当前Q形状
            kv_shape = [1, 1, self.num_key_value_heads, self.head_dim]# 当前K/V形状

            # 创建MLIR导入器（包含历史KV输入）
            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],# 输入
                [input_shape, kv_shape, kv_shape],# 输出
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],# 输入数据类型
                weight_file=weight_file)

            # 类型和位置简写
            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            # 定义输入操作符（包含历史KV）
            in0_op = block_mlir.create_input_op(L("input_states"), 0)# 当前状态
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)# 当前位置ID
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)# 注意力掩码
            in3_op = block_mlir.create_input_op(L("history_k"), 3)# 历史K缓存
            in4_op = block_mlir.create_input_op(L("history_v"), 4)# 历史V缓存
            return_ops = []
            # ======== 输入归一化 ========
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # ======== Q/K/V投影 ========
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, q_dim], [1, 1, q_dim])
            # k_proj
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # v_proj
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # reshape q,k,v
            # 重塑形状
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output

            # Q/K归一化（特定模型）
            if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)

            # ======== 旋转位置编码 ========
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)# 当前K缓存
            return_ops.append(v_op)# 当前V缓存

            # ======== KV缓存拼接 ========
            # 将历史K与当前K拼接
            k_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in3_op, k_op],# 历史K + 当前K
                                axis=1,# 沿序列维度拼接
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            # 将历史V与当前V拼接
            v_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in4_op, v_op],# 历史V + 当前V
                                axis=1,
                                loc=L(v_proj + ".concat"),
                                ip=ip).output
            # ======= fattention =========
            # ======== 注意力机制 ========
            fa_op = top.FAttentionOp(T([1, 1, q_dim]),# 输出形状（单token）
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=1,# 当前序列长度=1
                                     mk=self.seq_length + 1,# 总序列长度=历史+当前
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output

            # ======== 输出投影和残差 ========
            o_op = self.linear(block_mlir, o_proj, fa_op, [q_dim, self.hidden_size], input_shape)
            if self.llm_type == LlmType.GEMMA3:
                o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            if self.llm_type == LlmType.MINICPM4:
                o_op = top.MulConstOp(T(input_shape),
                                      o_op,
                                      const_val=self.scale_depth / np.sqrt(self.num_layers),
                                      loc=L(o_proj + ".scale0"),
                                      ip=ip).output
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            # ======== MLP处理 ========
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            # 创建返回操作：新状态 + 更新后的KV缓存
            block_mlir.create_return_op([new_op] + return_ops)
            # 输出MLIR文件
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        ########## 带KV缓存的预填充模式（多轮对话） ##########
        def gen_block_with_kv():
            # Generate block with kv cache related operations
            name = f"block_{idx}"# 名称标识
            input_len = self.max_input_length# 输入长度
            input_shape = [1, input_len, self.hidden_size]# 输入形状
            id_shape = list(self.position_shape)# 位置ID形状
            max_kv_len = self.max_prefill_kv_length + self.max_input_length# 最大KV长度（历史+当前）
            mask_shape = [1, 1, self.max_input_length, max_kv_len] # 掩码形状（当前seq x 总seq）
            history_shape = [1, self.max_prefill_kv_length, self.num_key_value_heads, self.head_dim]# 历史KV形状

            # Q/K/V形状
            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]

            # 创建MLIR导入器（包含历史KV输入）
            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],# 输入
                [input_shape, kv_shape, kv_shape],# 输出
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],# 输入数据类型
                weight_file=weight_file)

            # 类型和位置简写
            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            # 定义输入操作符（多轮对话需要历史KV）
            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            in3_op = block_mlir.create_input_op(L("history_k"), 3)
            in4_op = block_mlir.create_input_op(L("history_v"), 4)
            return_ops = []

            # ======== 输入归一化 ========
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # ======== Q/K/V投影 ========
            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, q_dim],
                               [1, input_len, q_dim])
            # k_proj
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])
            # v_proj
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])
            # reshape q,k,v
            # 重塑形状
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output

            # Q/K归一化（特定模型）
            if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)

            # rotary cos/sin
            # ======== 旋转位置编码 ========
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ====== kv concat ========
            # ======== KV缓存拼接 ========
            # 将历史K与当前K拼接
            k_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in3_op, k_op],# 历史K + 当前K
                                axis=1,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            # 将历史V与当前V拼接
            v_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in4_op, v_op],# 历史V + 当前V
                                axis=1,
                                loc=L(v_proj + ".concat"),
                                ip=ip).output
            # ======== 注意力机制 ========
            fa_op = top.FAttentionOp(T([1, input_len, q_dim]),# 输出形状（整个序列）
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=input_len,# 当前序列长度
                                     mk=max_kv_len,# 总序列长度（历史+当前）
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            # ======== 输出投影和残差 ========
            o_op = self.linear(block_mlir, o_proj, fa_op, [q_dim, self.hidden_size], input_shape)
            if self.llm_type == LlmType.GEMMA3:
                o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            if self.llm_type == LlmType.MINICPM4:
                o_op = top.MulConstOp(T(input_shape),
                                      o_op,
                                      const_val=self.scale_depth / np.sqrt(self.num_layers),
                                      loc=L(o_proj + ".scale0"),
                                      ip=ip).output
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            # ======== MLP处理 ========
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            # 创建返回操作：新状态 + 更新后的KV缓存
            block_mlir.create_return_op([new_op] + return_ops)
            # 输出MLIR文件
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        ########## 执行生成逻辑 ##########
        if self.use_block_with_kv:# 多轮对话模式
            gen_block_with_kv()
        else:# 单轮对话模式
            gen_block()
        gen_block_cache()# 总是生成解码模式

    # 生成ViT模型的MLIR（暂未实现）
    def gen_vit_mlir(self):
        pass

    # ============= compile all code =============
    def add_task(self, command: list[str], log_file: str):
        """
        添加编译任务到任务列表

        参数:
            command: 要执行的命令列表
            log_file: 日志文件路径
        """
        # 将输出重定向添加到命令末尾
        command.append(f"> {log_file}\n")
        # 将命令列表转换为单个命令字符串
        cmd = ' '.join(command)
        # 添加到任务列表
        self.commands.append(cmd)

    def run_command(self, command):
        """
        执行单个命令并处理结果

        参数:
            command: 要执行的命令列表
        """
        GREEN_COLOR = "\033[92m"    # 绿色文本
        RED_COLOR = "\033[91m"      # 红色文本
        RESET_COLOR = "\033[0m"     # 重置颜色
        try:
            # 打印绿色提示信息
            print(f"{GREEN_COLOR}Executing command: \n{' '.join(command)}{RESET_COLOR}"
                  )  # Print the command in green
            # 执行命令并检查返回码
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            # 打印红色错误信息
            print(f"{RED_COLOR}Error: Command failed with return code {e.returncode}{RESET_COLOR}")
            print(f"{RED_COLOR}Failed command: {' '.join(command)}{RESET_COLOR}")
            # 退出程序并返回相同的错误码
            sys.exit(e.returncode)

    def execute_tasks(self):
        """并行执行所有任务"""
        task_file = "task.txt"
        # 将所有任务写入文件
        with open(task_file, "w") as f:
            f.writelines(self.commands)
        self.commands.clear()# 清空任务列表

        # 构建并行执行命令
        parallel_cmd = [
            "parallel",                 # GNU parallel工具
            f"-j {self.max_workers}",   # 最大并行任务数
            "--halt now,fail=1",        # 任何任务失败立即停止
            "--progress",               # 显示进度信息
            f"--joblog {task_file}.log",# 任务日志文件
            f"< {task_file}"            # 从文件读取任务
        ]
        # 执行并行命令
        self.run_command(['bash', '-c', ' '.join(parallel_cmd)])

    def compile_embedding(self):
        """编译嵌入层模型"""
        name = "embedding"
        # 检查模型是否已存在
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        # 构建部署参数
        deploy_args = [
            'model_deploy.py',              # 模型部署工具
            f'--mlir {name}.mlir',          # MLIR输入文件
            f'--quantize {self.half_precision_quantize}',# 量化模式
            '--quant_input',                # 量化输入
            '--quant_output',               # 量化输出
            f'--chip {self.chip}',          # 目标芯片
            f'--num_core {self.num_core}',  # 使用核心数
            f'--num_device {self.num_device}',# 使用设备数
            f'--model {name}.bmodel'        # 输出模型文件
        ]
        # 调试模式添加额外参数
        if self.debug:
            deploy_args.append('--debug')
        # 添加任务
        self.add_task(deploy_args, f"{name}.log")

    def compile_embedding_cache(self):
        """编译缓存嵌入层模型"""
        name = "embedding_cache"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input', '--quant_output', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.add_task(deploy_args, f"{name}.log")

    def compile_lm_head(self):
        """编译语言模型头部"""
        name = "lm_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input',# 只量化输入（输出不量化）
            f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.add_task(deploy_args, f"{name}.log")

    def compile_greedy_head(self):
        """编译贪婪解码头部"""
        name = "greedy_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        # 贪婪解码头部不需要量化
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.add_task(deploy_args, f"{name}.log")

    def compile_sample_head(self):
        """编译采样解码头部"""
        name = "sample_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        # 采样解码头部不需要量化
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.add_task(deploy_args, f"{name}.log")

    def compile_block(self, layer_id):
        """编译Transformer块（完整序列处理）"""
        name = f"block_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            'model_deploy.py',
            f'--mlir {name}.mlir',
            f'--quantize {self.quantize}',# 量化模式
            f'--q_group_size {self.q_group_size}',# 量化组大小
            '--quant_input', '--quant_output',
            f'--chip {self.chip}',
            f'--num_core {self.num_core}',
            f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        # 添加可选参数
        if self.high_precision:
            deploy_args.append('--high_precision')# 高精度模式
        if self.symmetric:
            deploy_args.append('--q_symmetric')# 对称量化
        if self.dynamic:
            deploy_args.append('--dynamic')# 动态形状支持
        if self.debug:
            deploy_args.append('--debug')
        self.add_task(deploy_args, f"{name}.log")

    def compile_block_cache(self, layer_id):
        """编译Transformer块（缓存处理）"""
        name = f"block_cache_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.quantize}',
            f'--q_group_size {self.q_group_size}', '--quant_input', '--quant_output',
            f'--chip {self.chip}', '--addr_mode io_alone',# 特殊的地址模式
            f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        if self.debug:
            deploy_args.append('--debug')
        self.add_task(deploy_args, f"{name}.log")

    def compile_vit(self):
        """编译视觉Transformer模型"""
        if not self.do_vit:# 检查是否需要编译ViT
            return
        name = "vit"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        # ViT模型的特殊量化处理
        if self.half_precision_quantize == 'bf16' and self.vit_f16_out_bf16:
            deploy_args.append('--quantize f16')# 使用FP16量化
            deploy_args.append('--quant_output_bf16')# 输出转换为BF16
        else:
            deploy_args.append(f'--quantize {self.half_precision_quantize}')
            deploy_args.append('--quant_output')# 量化输出
        if self.high_precision:
            deploy_args.append('--high_precision')
        self.add_task(deploy_args, f"{name}.log")

    def combine(self):
        """合并所有编译的bmodel文件"""
        bmodel_list = []# 要合并的模型列表
        total_bytes = 0# 估计的总大小

        # 添加所有Transformer块
        for i in range(self.num_layers):
            bmodel_list = bmodel_list + [f"block_{i}.bmodel", f"block_cache_{i}.bmodel"]
            total_bytes += os.path.getsize("block_0.bmodel")
        # 添加嵌入层
        if not self.embedding_disk:
            bmodel_list += ['embedding.bmodel', 'embedding_cache.bmodel']
            total_bytes += os.path.getsize("embedding.bmodel")
        # 添加解码头部
        if not self.lmhead_with_topk:
            bmodel_list += ["greedy_head.bmodel", "sample_head.bmodel"]
        # 添加ViT模型
        if self.do_vit:
            bmodel_list += ["vit.bmodel"]
            total_bytes += os.path.getsize("vit.bmodel")
        # 添加语言模型头部
        bmodel_list += ["lm_head.bmodel"]

        total_bytes += os.path.getsize("lm_head.bmodel")

        # 构建合并命令
        combine_args = ['model_tool',# 模型工具
                        '--combine',# 合并模式
                        ' '.join(bmodel_list),# 要合并的文件列表
                        '-o', self.out_bmodel]# 输出文件
        # 执行合并命令
        self.run_command(['bash', '-c', ' '.join(combine_args)])
        # 获取合并后模型大小
        bmodel_size = os.path.getsize(self.out_bmodel)
        print(f"Combined bmodel size: {bmodel_size / (1024.0 ** 3)} GB")

        # 检查合并后模型大小是否合理
        if bmodel_size > total_bytes * 1.2:
            raise RuntimeError("Combined bmodel size is too large, please check the model.")

        # 生成模型信息日志
        get_info_args = ['model_tool', '--info', self.out_bmodel, '> ../model.log']
        self.run_command(['bash', '-c', ' '.join(get_info_args)])

    def compile_all(self):
        """编译所有组件"""
        # 编译ViT（如果需要）
        if self.do_vit:
            self.compile_vit()

        # 编译嵌入层（如果权重未存储在磁盘）
        if not self.embedding_disk:
            self.compile_embedding()
            self.compile_embedding_cache()

        # 编译语言模型头部
        self.compile_lm_head()

        # 编译解码头部（如果未使用TopK）
        if not self.lmhead_with_topk:
            self.compile_greedy_head()
            self.compile_sample_head()

        # 编译所有Transformer块
        for i in range(self.num_layers):
            self.compile_block(i)# 完整序列块
            self.compile_block_cache(i)# 缓存块

        # 执行所有编译任务
        self.execute_tasks()

        # 合并所有bmodel文件
        self.combine()

        # 清理临时文件（非调试模式）
        if not self.debug:
            # 删除所有.npz权重文件
            for npz_file in os.listdir():
                if os.path.splitext(npz_file)[-1] == '.npz':
                    os.remove(npz_file)
