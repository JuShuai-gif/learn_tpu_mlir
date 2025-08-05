# Copyright (C) 2025 Sophgo Technologies In"  All" rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
import math


class FAttention:
    """
    用于估算 Transformer Attention 模块的 FLOPs 和内存访问量（Bytes）。
    区分 prefill（批量填充阶段，通常是推理首轮处理长序列） 和 decode（解码阶段，增量生成）两种情况。
    """
    # prefill:
    # [B, NUM_Q_HEAD, L, HEAD_DIM] @ [B, NUM_Q_HEAD, HEAD_DIM, L] => [B, NUM_Q_HEAD, L, L]
    # [B, NUM_Q_HEAD, L, L] @ [B, NUM_Q_HEAD, L, HEAD_DIM] => [B, NUM_Q_HEAD, L, HEAD_DIM]
    # decode:
    # [B, NUM_Q_HEAD, 1, HEAD_DIM] @ [B, NUM_Q_HEAD, HEAD_DIM, L+1] => [B, NUM_Q_HEAD, 1, L+1]
    # [B, NUM_Q_HEAD, 1, L+1] @ [B, NUM_Q_HEAD, L+1, HEAD_DIM] => [B, NUM_Q_HEAD, 1, HEAD_DIM]
    def __init__(self, NUM_Q_HEAD, SEQ, HEAD_DIM, prefill: bool = True):
        """
        :param NUM_Q_HEAD: 注意力头数量
        :param SEQ: 序列长度 L
        :param HEAD_DIM: 每个注意力头的维度
        :param prefill: 是否是 prefill 阶段（True: 首轮计算，False: 增量解码）
        """
        self.B = 1 * NUM_Q_HEAD# 批大小*注意力头数（此处假设batch=1）
        self.M0 = SEQ if prefill else 1# 第一阶段左矩阵行数 (prefill: L, decode: 1)
        self.K0 = HEAD_DIM# 第一阶段公共维度
        self.N0 = SEQ if prefill else SEQ + 1# 第一阶段右矩阵列数 (prefill: L, decode: L+1)
        self.M1 = SEQ if prefill else 1# 第二阶段左矩阵行数 (prefill: L, decode: 1)
        self.K1 = SEQ if prefill else SEQ + 1# 第二阶段公共维度
        self.N1 = HEAD_DIM# 第二阶段右矩阵列数

    def get_flops(self):
        """
        计算 Attention 两次矩阵乘法的 FLOPs（浮点运算数）
        FLOPs公式: 2 * B * (M*K*N)
        """
        self.flops = 2 * self.B * (self.M0 * self.K0 * self.N0 + self.M1 * self.K1 * self.N1)
        return self.flops

    def get_bytes(self, quantize_type, group_size):
        """
        估算内存访问字节数 (Bytes)
        :param quantize_type: 量化类型（暂未用到）
        :param group_size: 分组大小（暂未用到）
        """
        # 计算输入输出的大小（假定fp16，1个数2字节）
        self.bytes = 2 * self.B * (self.M0 * self.K0 +# Q输入
                                   self.K0 * self.N0 +# K输入
                                   self.K1 * self.N1 +# V输入
                                   self.M1 * self.N1) + 2 * self.M0 * self.N0# 输出   注意力权重矩阵
        return self.bytes


class MatMul:
    """
    通用矩阵乘法 FLOPs 和 Bytes 计算工具
    """
    def __init__(self, L: list, R: list):
        """
        :param L: 左矩阵的shape列表，如 [..., M, K]
        :param R: 右矩阵的shape列表，如 [..., K, N]
        """
        assert (len(L) >= len(R))# 左矩阵维度要大于等于右矩阵
        self.B = 1
        # 如果是批量矩阵乘法 (len(L) > 2)，则取前面维度的乘积作为批大小
        if len(L) > 2:
            self.B = math.prod(L[:-2])
        self.M = L[-2]# 左矩阵行数
        self.K = L[-1]# 公共维度
        self.N = R[-1]# 右矩阵列数
        # IO字节: 输入(L + 输出(M*N))，假定每个元素2字节（f16）
        self.io_bytes = 2 * self.B * (self.M * self.K + self.M * self.N)

    def get_flops(self):
        """
        计算FLOPs = 2 * B * M * K * N
        """
        self.flops = 2 * self.B * self.M * self.K * self.N
        return self.flops

    def get_bytes(self, quantize_type="f16", group_size=64):
        """
        根据量化方式计算内存访问量
        :param quantize_type: 权重量化类型
            - "f16", "bf16"     : 全精度16bit
            - "w8f16", "w8bf16" : 权重量化到8bit，激活为16bit
            - "w4f16", "w4bf16" : 权重量化到4bit，激活为16bit
        :param group_size: 量化分组大小 (用于存储scale、zero_point等参数)
        """
        if quantize_type == "f16" or quantize_type == "bf16":
            # 输入输出为16bit, 权重16bit
            self.bytes_a16 = self.io_bytes + 2 * self.K * self.N
            return self.bytes_a16
        elif quantize_type == "w8f16" or quantize_type == "w8bf16":
            # 权重8bit，额外存储量化参数(3*K*N/group_size)
            self.bytes_w8a16 = self.io_bytes + self.K * self.N + 3 * self.K * self.N / group_size
            return self.bytes_w8a16
        elif quantize_type == "w4f16" or quantize_type == "w4bf16":
            # 权重4bit，额外存储量化参数(3*K*N/group_size)
            self.bytes_w4a16 = self.io_bytes + 0.5 * self.K * self.N + 3 * self.K * self.N / group_size
            return self.bytes_w4a16
        else:
            raise ValueError(f"Unsupported quantize type: {quantize_type}")


class LlmProfiler:
    """大型语言模型性能分析器，用于预估模型在目标硬件上的性能指标"""
    def __init__(self, args, config):
        # user config
        # 用户配置参数
        self.seq_length = args.seq_length# 序列长度
        self.quantize = args.quantize# 量化方式（如'int4', 'int8'）
        self.group_size = args.group_size# 量化分组大小
        self.chip = args.chip# 目标芯片型号（如'bm1684x'）
        self.num_device = args.num_device# 设备数量（多卡）
        self.num_core = args.num_core# 核心数量（多核）
        # LM头部是否量化，默认与主体相同
        self.lmhead_quantize = self.quantize if args.quant_lmhead else 'f16'
        # 模型名称（从路径提取）
        self.model_name = os.path.basename(args.model_path.rstrip('/')).lower()
        # 性能分析名称（用于标识分析结果）
        self.profile_name = f"{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}"
        # chip config
        # 根据芯片类型设置硬件特性参数
        if self.chip == "bm1684x":
            self.tflops = 16.               # 芯片的峰值TFLOPS
            self.dma_bw = 64.               # DMA带宽（GB/s）
            self.p2p_bw = 3.                # 点对点通信带宽（GB/s）
            self.num_core = 1               # 核心数固定为1
            self.prefill_mac_util = 0.55    # prefill阶段计算单元利用率
            self.prefill_ddr_util = 0.3     # prefill阶段内存带宽利用率
            self.decode_mac_util = 0.2      # decode阶段计算单元利用率
            self.decode_ddr_util = 0.85     # decode阶段内存带宽利用率
            self.tpu_freq = 950             # TPU频率（MHz）
            self.profile_name += f"_{self.num_device}dev"# 添加设备数量到分析名称
        elif self.chip == "bm1688":
            self.tflops = 3.6
            self.dma_bw = 32.
            self.num_device = 1# 单设备
            self.prefill_mac_util = 0.5
            self.prefill_ddr_util = 0.1
            self.decode_mac_util = 0.1
            self.decode_ddr_util = 0.8
            self.tpu_freq = 900
            self.profile_name += f"_{self.num_core}core"# 添加核心数到分析名称
        elif self.chip == "cv186x":
            self.tflops = 1.8
            self.dma_bw = 24.
            self.num_core = 1
            self.num_device = 1
            self.prefill_mac_util = 0.5
            self.prefill_ddr_util = 0.1
            self.decode_mac_util = 0.1
            self.decode_ddr_util = 0.8
            self.tpu_freq = 750
            self.profile_name += f"_{self.num_core}core"
        else:
            raise ValueError(f"Unsupported chip type: {args.chip}")
        # 使用用户指定的TPU频率（如果提供）
        self.tpu_freq = args.tpu_freq if args.tpu_freq is not None else self.tpu_freq
        self.profile_name += f"_{self.tpu_freq}MHz"# 添加频率到分析名称
        # model config
        # 从模型配置中提取关键参数
        self.hidden_size = config.hidden_size# 隐藏层维度
        self.interm_size = config.intermediate_size# MLP中间层维度
        self.num_attn_heads = config.num_attention_heads# 注意力头数
        self.num_kv_heads = config.num_key_value_heads# K/V头数（用于分组查询）
        self.num_layers = config.num_hidden_layers# 模型层数
        self.vocab_size = config.vocab_size# 词汇表大小
        # 计算每个注意力头的维度
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_attn_heads)
        self.q_dim = self.num_attn_heads * self.head_dim# Q向量总维度
        self.kv_dim = self.num_kv_heads * self.head_dim# K/V向量总维度
        # 计算每个K/V头对应的查询头数（分组查询）
        self.kv_tile = self.num_attn_heads // self.num_kv_heads
        # 是否共享词嵌入权重
        self.tie_word_embeddings = config.tie_word_embeddings
        # 获取量化配置（如果存在）
        self.quantization_config = getattr(config, "quantization_config", None)
        if self.quantization_config:
            self.group_size = self.quantization_config["group_size"]# 覆盖分组大小

    def get_tiu_time(self, total_flops, mac_util, tflops=0.):
        """计算张量计算单元(TIU)执行时间"""
        tflops = self.tflops if tflops == 0. else tflops# 使用默认或指定算力
        tpu_peak_flops = tflops * 1024 * 1e9# 将TFLOPS转换为FLOPS
        # 计算时间：总操作数 / (峰值性能 * 频率因子) / 利用率 * 1000（毫秒）
        tiu_time_ms = total_flops / tpu_peak_flops / (self.tpu_freq / 1000) / mac_util * 1000
        return tiu_time_ms

    def get_dma_time(self, total_bytes, ddr_util):
        """计算直接内存访问(DMA)时间"""
        gbytes_per_dev = total_bytes / 2**30# 字节转GB
        # 计算时间：数据量 / 带宽 / 频率因子 / 利用率 * 1000（毫秒）
        dma_time_ms = gbytes_per_dev / self.dma_bw / (self.tpu_freq / 1000) / ddr_util * 1000
        return dma_time_ms

    def get_allreduce_time(self, prefill: bool = True):
        """计算AllReduce通信时间（多设备场景）"""
        allreduce_num = self.num_layers * 2# 每层有两次AllReduce
        if prefill:
            # Prefill阶段（处理整个序列）
            bf16_size = 2# BF16数据类型占2字节
            # 计算环形AllReduce的数据比例
            ring_data_ratio = (self.num_device - 1) * 2 / self.num_device
            # 计算总数据量（GB）
            gbytes = self.seq_length * self.hidden_size * bf16_size * ring_data_ratio * allreduce_num / 2**30
            # 点对点通信时间
            p2p_time_ms = gbytes / self.p2p_bw * self.tpu_freq
            # 加法操作时间
            add_time_ms = (gbytes * 3 / self.dma_bw) / self.prefill_ddr_util * self.tpu_freq
            allreduce_time_ms = p2p_time_ms + add_time_ms
            return allreduce_time_ms
        else:
            # Decode阶段（逐个token处理）
            # 使用经验值估算AllReduce时间
            time_per_allreduce = 0
            if self.num_device == 2:
                time_per_allreduce = 0.12
            elif self.num_device == 4:
                time_per_allreduce = 0.15
            elif self.num_device == 8:
                time_per_allreduce = 0.2
            elif self.num_device == 1:
                return 0
            allreduce_time_ms = time_per_allreduce * allreduce_num
            return allreduce_time_ms

    def get_pcie_interrupt_time(self, pcie_avg_ms):
        """计算PCIe中断总时间（多设备场景）"""
        pcie_time_ms = pcie_avg_ms * self.num_layers * 2
        return pcie_time_ms

    def _analyze_prefill(self):
        """分析Prefill阶段（处理整个输入序列）的性能"""
        # 定义Prefill阶段的各计算模块
        self.prefill_stage = [
            # attn
            # 注意力层中的Q投影
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.q_dim / self.num_device]),
            # 注意力层中的K投影
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.kv_dim / self.num_device]),
            # 注意力层中的V投影
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.kv_dim / self.num_device]),
            # 注意力计算（FlashAttention）
            FAttention(self.num_attn_heads / self.num_device,
                       self.seq_length,
                       self.head_dim,
                       prefill=True),
            # 注意力输出投影
            MatMul([1, self.seq_length, self.q_dim / self.num_device],
                   [self.q_dim / self.num_device, self.hidden_size]),
            # mlp
            # MLP第一层（升维）
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            # MLP第二层（降维）
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            # MLP输出层
            MatMul([1, self.seq_length, self.interm_size / self.num_device],
                   [self.interm_size / self.num_device, self.hidden_size])
        ]
        # LM头部（输出层）
        lm_head = MatMul([1, self.hidden_size / self.num_device],
                         [self.hidden_size / self.num_device, self.vocab_size])
        # 计算LM头部时间（取DMA和TIU中的最大值）
        lmhead_time = max(
            self.get_dma_time(lm_head.get_bytes(self.lmhead_quantize), self.prefill_ddr_util),
            self.get_tiu_time(lm_head.get_flops(), self.prefill_mac_util, tflops=self.tflops / 4))
        # 计算整个模型的FLOPs和内存访问量
        block_flops = sum([op.get_flops() for op in self.prefill_stage]) * self.num_layers
        block_bytes = sum(
            [op.get_bytes(self.quantize, self.group_size)
             for op in self.prefill_stage]) * self.num_layers
        # 计算每层时间（DMA和TIU并行，取最大值）
        block_time = sum([
            max(
                self.get_dma_time(op.get_bytes(self.quantize, self.group_size),
                                  self.prefill_ddr_util),
                self.get_tiu_time(op.get_flops(), self.prefill_mac_util))
            for op in self.prefill_stage
        ]) * self.num_layers
        # 汇总Prefill阶段指标
        self.prefill_flops = block_flops + lm_head.get_flops()
        self.prefill_bytes = block_bytes + lm_head.get_bytes(self.lmhead_quantize)

        # 理论最佳时间（利用率100%）
        self.prefill_tiu_theo_time = self.get_tiu_time(self.prefill_flops, mac_util=1.)
        self.prefill_dma_theo_time = self.get_dma_time(self.prefill_bytes, ddr_util=1.)

        # 实际预估时间（考虑利用率）
        self.prefill_tiu_time = self.get_tiu_time(self.prefill_flops, self.prefill_mac_util)
        self.prefill_dma_time = self.get_dma_time(self.prefill_bytes, self.prefill_ddr_util)
        # AllReduce通信时间
        self.prefill_allreduce_time = self.get_allreduce_time(prefill=True)
        # 总时间 = 计算时间 + LM头时间 + 通信时间
        self.prefill_total_time = block_time + lmhead_time + self.prefill_allreduce_time

    def _analyze_decode(self):
        """分析Decode阶段（逐个token生成）的性能"""
        # 定义Decode阶段的各计算模块（序列长度为1）
        self.decode_stage = [
            # attn
            # 注意力层中的Q投影（单个token）
            MatMul([1, 1, self.hidden_size], [self.hidden_size, self.q_dim / self.num_device]),
            # 注意力层中的K投影（单个token）
            MatMul([1, 1, self.hidden_size], [self.hidden_size, self.kv_dim / self.num_device]),
            # 注意力层中的V投影（单个token）
            MatMul([1, 1, self.hidden_size], [self.hidden_size, self.kv_dim / self.num_device]),
            # 注意力计算（使用KV缓存）
            FAttention(self.num_attn_heads / self.num_device,
                       self.seq_length,
                       self.head_dim,
                       prefill=False),
            # 注意力输出投影
            MatMul([1, 1, self.q_dim / self.num_device],
                   [self.q_dim / self.num_device, self.hidden_size]),
            # mlp
            # MLP第一层
            MatMul([1, 1, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            # MLP第二层
            MatMul([1, 1, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            # MLP输出层
            MatMul([1, 1, self.interm_size / self.num_device],
                   [self.interm_size / self.num_device, self.hidden_size])
        ]
        # LM头部（输出层）
        lm_head = MatMul([1, self.hidden_size / self.num_device],
                         [self.hidden_size / self.num_device, self.vocab_size])
        # 计算LM头部时间
        lmhead_time = max(
            self.get_dma_time(lm_head.get_bytes(self.lmhead_quantize), self.decode_ddr_util),
            self.get_tiu_time(lm_head.get_flops(), self.decode_mac_util, tflops=self.tflops / 4))
        # 计算整个模型的FLOPs和内存访问量
        block_flops = sum([op.get_flops() for op in self.decode_stage]) * self.num_layers

        block_bytes = sum(
            [op.get_bytes(self.quantize, self.group_size)
             for op in self.decode_stage]) * self.num_layers
        # 计算每层时间（考虑decode阶段TIU利用率较低）
        block_time = sum([
            max(
                self.get_dma_time(op.get_bytes(self.quantize, self.group_size),
                                  self.decode_ddr_util),
                self.get_tiu_time(op.get_flops(), self.decode_mac_util, tflops=self.tflops / 4))
            for op in self.decode_stage
        ]) * self.num_layers
        # 汇总Decode阶段指标
        self.decode_flops = block_flops + lm_head.get_flops()
        self.decode_bytes = block_bytes + lm_head.get_bytes(self.lmhead_quantize)
        # 理论最佳时间
        self.decode_tiu_theo_time = self.get_tiu_time(self.decode_flops, mac_util=1.)
        self.decode_dma_theo_time = self.get_dma_time(self.decode_bytes, ddr_util=1.)
        # 实际预估时间
        self.decode_tiu_time = self.get_tiu_time(self.decode_flops, self.decode_mac_util)
        self.decode_dma_time = self.get_dma_time(self.decode_bytes, self.decode_ddr_util)
        # 通信和中断时间
        self.decode_allreduce_time = self.get_allreduce_time(prefill=False)
        self.decode_pcie_time = self.get_pcie_interrupt_time(pcie_avg_ms=0.05)
        # 总时间 = 计算时间 + LM头时间 + 通信时间 + 中断时间
        self.decode_total_time = block_time + lmhead_time + self.decode_allreduce_time + self.decode_pcie_time

    def _analyze_mem_usage(self):
        """分析模型内存使用情况"""
        # LM头部内存
        lm_head = MatMul([1, self.hidden_size / self.num_device],
                         [self.hidden_size / self.num_device, self.vocab_size])
        # 计算权重内存（排除输入输出）
        weight_bytes = sum([
                           op.get_bytes(self.quantize, self.group_size) - op.io_bytes \
                           if isinstance(op, MatMul) else 0 for op in self.decode_stage
                       ]) * self.num_layers + lm_head.get_bytes() - lm_head.io_bytes

        # 如果不共享词嵌入，添加额外权重
        if not self.tie_word_embeddings:
            weight_bytes += lm_head.get_bytes(self.lmhead_quantize) - lm_head.io_bytes

        # KV缓存内存（bf16格式，2字节/元素）
        kv_cache_bytes = self.seq_length * self.kv_dim * 2 * 2
        instruct_bytes = 0.# 指令内存（预留）
        runtime_bytes = 0.# 运行时内存（预留）
        sys_usage = 78 * 2**20# 系统预留内存（78MB）

        # 总内存使用 = 权重 + KV缓存 + 系统预留
        self.memory_usage = weight_bytes + instruct_bytes + runtime_bytes + kv_cache_bytes + sys_usage

    def analyze(self):
        """执行完整的性能分析并打印结果"""
        self._analyze_prefill()     # 分析Prefill阶段
        self._analyze_decode()      # 分析Decode阶段
        self._analyze_mem_usage()   # 分析内存使用

        # 打印模型配置
        print(f"\n=== {self.profile_name} ===")
        print(f"Model Config:")
        print(f'  hidden_size: {self.hidden_size}')
        print(f'  num_layers: {self.num_layers}')
        print(f'  num_attn_heads: {self.num_attn_heads}')
        print(f'  num_kv_heads: {self.num_kv_heads}')
        print(f'  intermediate_size: {self.interm_size}'),
        print(f'  vocab_size: {self.vocab_size}\n')

        # 打印Prefill阶段结果
        print("Prefill:")
        print(f"  Total Flops: {self.prefill_flops / 1e9:.3f} GFLOPs")
        print(f"  Total Bytes: {self.prefill_bytes / 2**20:.3f} MiB")
        print(f"  Total Time: {self.prefill_total_time:.3f} ms")
        print(f"  TPU Theo Time: {self.prefill_tiu_theo_time:.3f} ms")
        print(f"  DDR Theo Time: {self.prefill_dma_theo_time:.3f} ms")
        print(f"  TPU Time: {self.prefill_tiu_time:.3f} ms")
        print(f"  DDR Time: {self.prefill_dma_time:.3f} ms\n")

        # 打印Decode阶段结果
        print("Decode:")
        print(f"  Total Flops: {self.decode_flops / 1e9:.3f} GFLOPs")
        print(f"  Total Bytes: {self.decode_bytes / 2**20:.3f} MiB")
        print(f"  Total Time: {self.decode_total_time:.3f} ms")
        print(f"  TPU Theo Time: {self.decode_tiu_theo_time:.3f} ms")
        print(f"  DDR Theo Time: {self.decode_dma_theo_time:.3f} ms")
        print(f"  TPU Time: {self.decode_tiu_time:.3f} ms")
        print(f"  DDR Time: {self.decode_dma_time:.3f} ms\n")

        # 打印关键性能指标
        print(f"FTL: {self.prefill_total_time / 1000:.3} s")        # 首次令牌延迟
        print(f"TPS: {1000 / self.decode_total_time:.3f} token/s")  # 每秒生成令牌数
        print(f"Mem: {self.memory_usage / 2**20:.3f} MiB\n")        # 内存使用量
