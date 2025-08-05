# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


class ModelConfig:
    """
    模型配置类，用于描述大型语言模型（LLM）的结构参数。
    注意：此类仅存储参数的字段名（字符串），并不是参数的具体数值。
    这些字段名通常用于在权重文件或配置文件中查找对应的参数。
    """
    def __init__(self,
                 num_attention_heads: str = 'num_attention_heads',  # 注意力头数量字段
                 num_hidden_layers: str = 'num_hidden_layers',      # Transformer 层数字段
                 num_key_value_heads: str = 'num_key_value_heads',  # KV缓存头数量字段（部分模型使用多组KV头）
                 hidden_size: str = 'hidden_size',                  # 隐藏层维度字段
                 vocab_size: str = 'vocab_size',                    # 词表大小字段
                 intermediate_size: str = 'intermediate_size',      # MLP中间层维度字段
                 rope_theta: str = "rope_theta",                    # RoPE位置编码参数字段
                 rms_norm_eps: str = "rms_norm_eps",                # RMSNorm层的epsilon值字段
                 hidden_act: str = "hidden_act",                    # 激活函数类型字段
                 quantization_config: str = "quantization_config"): # 量化配置字段
        # 将参数名保存为对象属性
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.quantization_config = quantization_config


# only for llm, not for vlm
# 仅适用于纯语言模型（LLM），不适用于多模态（VLM）模型
class LlmType:
    """
    不同LLM模型类型的标识枚举（字符串常量）
    """
    QWEN2 = "qwen2"
    LLAMA = "llama"
    QWEN3 = "qwen3"
    CHATGLM3 = "chatglm"
    GEMMA3 = "gemma3_text"
    MINICPM4 = "minicpm"


class ActType:
    """
    激活函数类型枚举
    """
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SILU = "silu"
    TANH = "tanh"
    QUICK_GELU = "quick_gelu"
    GELU_PYTORCH_TANH = "gelu_pytorch_tanh" # PyTorch版本的GELU(tanh近似)


class WeightType:
    """
    权重归一化类型枚举
    """
    NORMAL = "Normal"       # 无归一化（普通权重）
    RMS_NORM = "RMSNorm"    # 使用RMSNorm归一化
    LAYER_NORM = "LayerNorm"# 使用LayerNorm归一化


class LlmList:
    """
    模型权重名称列表（用于组织和访问权重字典时的关键字）
    """
    LAYERS = "LAYERS"           # 整个Transformer层集合
    EMBEDING = "EMBEDING"       # 词嵌入层
    # =========== in layers ==========
    INPUT_LN = "INPUT_LN"       # 输入归一化层
    Q_PROJ = "Q_PROJ"           # Q投影权重
    Q_NORM = "Q_NORM"           # Q归一化层（qwen3特有）
    K_PROJ = "K_PROJ"           # K投影权重
    K_NORM = "K_NORM"           # K归一化层（qwen3特有）
    V_PROJ = "V_PROJ"           # V投影权重
    O_PROJ = "O_PROJ"           # 输出投影权重
    QKV_WB = "QKV_WB"           # QKV合并权重（chatglm结构）
    ATT_D = "ATT_D"             # 注意力偏置（chatglm结构）
    POST_ATTN_LN = "POST_ATTN_LN"# 注意力后归一化层
    # MLP相关
    PRE_MLP_LN = "PRE_MLP_LN"   # MLP前归一化层（gemma3结构）
    POST_MLP_LN = "POST_MLP_LN" # MLP后归一化层（gemma3结构）
    MLP_GATE = "MLP_GATE"       # MLP门控层
    MLP_UP = "MLP_UP"           # MLP上投影
    MLP_DOWN = "MLP_DOWN"       # MLP下投影
    # ===============================
    NORM = "NORM"               # 最终归一化层
    LMHEAD = "LMHEAD"           # 语言建模输出头（输出词表logits）


class ModelInfo:
    """
    模型信息类，用于将模型的配置信息和权重信息绑定在一起
    """
    def __init__(self, config: ModelConfig, weights: map):
        """
        :param config: 模型配置对象
        :param weights: 模型权重映射（字典或其他结构）
        """
        self.config = config        # 模型结构配置信息
        self.weights = weights      # 模型权重数据


# qwen3/qwen2/llama
# 对 qwen3 / qwen2 / llama 系列模型的通用映射信息
COMMON_INFO = ModelInfo(
    ModelConfig(),
    weights={
        # Transformer 层集合的路径
        LlmList.LAYERS: "model.layers",
        # 词嵌入层（embedding）
        LlmList.EMBEDING: "model.embed_tokens",
        # ====== Transformer 每层内部模块的参数名称映射 ======
        LlmList.INPUT_LN: "input_layernorm",# 输入归一化层
        LlmList.Q_PROJ: "self_attn.q_proj",# Q投影权重
        LlmList.Q_NORM: "self_attn.q_norm",  # Q归一化（qwen3专有）
        LlmList.K_PROJ: "self_attn.k_proj",# K投影权重
        LlmList.K_NORM: "self_attn.k_norm",  # K归一化（qwen3专有）
        LlmList.V_PROJ: "self_attn.v_proj",# V投影权重
        LlmList.O_PROJ: "self_attn.o_proj",# 输出投影权重
        LlmList.POST_ATTN_LN: "post_attention_layernorm",# 注意力之后的归一化
        LlmList.MLP_GATE: "mlp.gate_proj",# MLP的门控层
        LlmList.MLP_UP: "mlp.up_proj",# MLP的上投影层
        LlmList.MLP_DOWN: "mlp.down_proj",# MLP的下投影层
        # ================================
        LlmList.NORM: "model.norm",# 最终归一化层
        LlmList.LMHEAD: "lm_head",# 输出logits的头部
    })

# 对 Phi3 模型的映射信息（结构和命名与 llama 不完全相同）
PHI3_INFO = ModelInfo(
    ModelConfig(),# 使用默认的模型配置字段名
    weights={
        LlmList.LAYERS: "model.layers",# Transformer 层集合
        LlmList.EMBEDING: "model.embed_tokens",# 词嵌入
        # ========= in layers =============
        # ========== Phi3 层内部结构 ==========
        LlmList.INPUT_LN: "input_layernorm",        # 输入归一化
        LlmList.QKV_WB: "self_attn.qkv_proj",       # 将Q、K、V合并在一起的权重
        LlmList.ATT_D: "self_attn.o_proj",          # 注意力输出层
        LlmList.POST_ATTN_LN: "post_attention_layernorm",# 注意力后归一化
        LlmList.MLP_UP: "mlp.gate_up_proj",         # MLP上投影 + 门控合并
        LlmList.MLP_DOWN: "mlp.down_proj",          # MLP下投影
        # ================================
        LlmList.NORM: "model.norm",                 # 最终归一化
        LlmList.LMHEAD: "lm_head",                  # 输出头
    })

# chatglm3
# 对 ChatGLM3 模型的映射信息
CHATGLM3_INFO = ModelInfo(
    ModelConfig(intermediate_size="ffn_hidden_size",# MLP中间层维度在该模型中字段名不同
                rms_norm_eps="layernorm_epsilon",# RMSNorm的epsilon字段名不同
                num_key_value_heads="multi_query_group_num",# KV组数参数名不同
                num_hidden_layers="num_layers"),# 层数参数名不同
    weights={
        LlmList.LAYERS: "transformer.encoder.layers",# 层集合
        LlmList.EMBEDING: "transformer.embedding.word_embeddings",# 词嵌入
        # ========= in layers =============
        # ========= ChatGLM3 层内部结构 =========
        LlmList.INPUT_LN: "input_layernorm",# 输入归一化
        LlmList.QKV_WB: "self_attention.query_key_value",# 合并的QKV权重
        LlmList.ATT_D: "self_attention.dense",# 注意力输出层
        LlmList.POST_ATTN_LN: "post_attention_layernorm",# 注意力后归一化
        LlmList.MLP_UP: "mlp.dense_h_to_4h",# MLP第一层（升维）
        LlmList.MLP_DOWN: "mlp.dense_4h_to_h",# MLP第二层（降维）
        # ================================
        LlmList.NORM: "transformer.encoder.final_layernorm",# 最终归一化层
        LlmList.LMHEAD: "transformer.output_layer",# 输出头
    })

# gemma3
# 对 Gemma3 模型的映射信息
GEMMA3_INFO = ModelInfo(
    ModelConfig(hidden_act="hidden_activation", ),# Gemma3 模型激活函数字段名不同
    weights={
        LlmList.LAYERS: "language_model.model.layers",# 层集合
        LlmList.EMBEDING: "language_model.model.embed_tokens",# 词嵌入
        # ========== Gemma3 层内部结构 =========
        LlmList.INPUT_LN: "input_layernorm",# 输入归一化
        LlmList.Q_PROJ: "self_attn.q_proj",# Q投影
        LlmList.Q_NORM: "self_attn.q_norm",# Q归一化
        LlmList.K_PROJ: "self_attn.k_proj",# K投影
        LlmList.K_NORM: "self_attn.k_norm",# K归一化
        LlmList.V_PROJ: "self_attn.v_proj",# V投影
        LlmList.O_PROJ: "self_attn.o_proj",# 输出投影
        LlmList.POST_ATTN_LN: "post_attention_layernorm",# 注意力后归一化
        LlmList.PRE_MLP_LN: "pre_feedforward_layernorm",# MLP前归一化（Gemma3特有）
        LlmList.POST_MLP_LN: "post_feedforward_layernorm",# MLP后归一化（Gemma3特有）
        LlmList.MLP_GATE: "mlp.gate_proj",# MLP门控
        LlmList.MLP_UP: "mlp.up_proj",# MLP上投影
        LlmList.MLP_DOWN: "mlp.down_proj",# MLP下投影
        # ================================
        LlmList.NORM: "language_model.model.norm",# 最终归一化
        LlmList.LMHEAD: "language_model.model.lm_head",# 输出头
    })
