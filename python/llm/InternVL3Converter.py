# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override

# 用于处理int64最大值的常量
int64_max = np.iinfo(np.int64).max


class InternVL3Converter(LlmConverter):
    """InternVL3模型转换器，专门处理视觉编码器和语言模型集成"""
    def __init__(self, args, config):
        super().__init__(args, config)
        # 启用视觉转换器(ViT)处理
        self.do_vit = True
        # modify weight info
        # 在权重名前添加"language_model."前缀，适配模型结构
        for i in ["LAYERS", "EMBEDING", "NORM", "LMHEAD"]:
            self.model_info.weights[i] = "language_model." + self.model_info.weights[i]
        # vision config
        # 视觉配置参数初始化
        self.vision_config = config.vision_config
        self.downsample_ratio = config.downsample_ratio# 特征下采样比例
        self.image_size = self.vision_config.image_size# 输入图像尺寸
        self.patch_size = self.vision_config.patch_size# 图像分块大小

        # 计算图像token数量：(图像尺寸/分块尺寸)^2 * 下采样比例^2
        self.num_image_token = int(
            (self.image_size // self.patch_size)**2 * (self.downsample_ratio**2))

        # ViT架构关键参数
        self.depth = self.vision_config.num_hidden_layers# Transformer层数
        self.vit_hidden_size = self.vision_config.hidden_size# 隐藏层维度
        self.vit_num_heads = self.vision_config.num_attention_heads# 注意力头数
        self.vit_head_dim = self.vit_hidden_size // self.vit_num_heads# 每头维度
        self.vit_intermediate_size = self.vision_config.intermediate_size# FFN层维度
        self.vit_ln_eps = self.vision_config.layer_norm_eps# LayerNorm epsilon值

    @override
    def load_pretrained(self, config):
        """加载预训练模型配置"""
        super().load_pretrained(config)
        # 保存语言模型配置信息
        self.llm_config = config.llm_config
        self.llm_type = self.llm_config.model_type

    def vision_block(self, vit_mlir, idx: int, in_op):
        """构建单个ViT模块的MLIR表示"""
        # 权重路径定义
        attn_qkv = f"vision_model.encoder.layers.{idx}.attn.qkv"
        attn_proj = f"vision_model.encoder.layers.{idx}.attn.proj"
        norm1 = f"vision_model.encoder.layers.{idx}.norm1"
        norm2 = f"vision_model.encoder.layers.{idx}.norm2"
        mlp = f"vision_model.encoder.layers.{idx}.mlp.fc"
        ls1 = f"vision_model.encoder.layers.{idx}.ls1"
        ls2 = f"vision_model.encoder.layers.{idx}.ls2"
        ip = vit_mlir.insert_point

        # 工具函数简化代码
        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)# 获取张量类型

        def L(name: str):
            return self.get_loc(name, vit_mlir)# 获取位置标记

        # ----------------------- 自注意力子层 -----------------------
        def vision_attention(in_op):
            """构建注意力机制的MLIR操作"""
            # 层归一化 (Pre-LN)
            weight_op0 = vit_mlir.create_weight_op(norm1 + ".weight", [1, 1, self.vit_hidden_size])
            weight_op1 = vit_mlir.create_weight_op(norm1 + ".bias", [1, 1, self.vit_hidden_size])
            norm1_op = top.LayerNormOp(T([1, 1025, self.vit_hidden_size]),
                                       in_op,
                                       weight_op0,
                                       weight_op1,
                                       axis=2,
                                       eps=self.vit_ln_eps,
                                       normalized_shape=[self.vit_hidden_size],
                                       loc=L(norm1),
                                       ip=ip).output
            # QKV全连接层 (合并实现)
            weight_op2 = vit_mlir.create_weight_op(attn_qkv + ".weight",
                                                   [self.vit_hidden_size, self.vit_hidden_size * 3])
            weight_op3 = vit_mlir.create_weight_op(attn_qkv + ".bias",
                                                   [1, 1, self.vit_hidden_size * 3])
            qkv_mm_op = top.MatMulOp(T([1, 1025, self.vit_hidden_size * 3]),
                                     norm1_op,
                                     weight_op2,
                                     weight_op3,
                                     loc=L(attn_qkv),
                                     ip=ip).output

            # 拆分QKV (形状: [batch, seq, 3, num_heads, head_dim])
            reshape_op = top.ReshapeOp(T([1, 1025, 3, self.vit_num_heads, self.vit_head_dim]),
                                       qkv_mm_op,
                                       shape=[1, 1025, 3, self.vit_num_heads, self.vit_head_dim],
                                       loc=L(attn_qkv + ".reshape"),
                                       ip=ip).output

            # 切分Q、K、V
            q_slice_op = top.SliceOp(T([1, 1025, 1, self.vit_num_heads, self.vit_head_dim]),
                                     reshape_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     offset=[0, 0, 0, 0, 0],
                                     steps=[1, 1, 1, 1, 1],
                                     ends=[int64_max, int64_max, 1, int64_max, int64_max],
                                     hasparamConvert_axes=[2],
                                     loc=L(attn_qkv + ".q.slice"),
                                     ip=ip).output
            k_slice_op = top.SliceOp(T([1, 1025, 1, self.vit_num_heads, self.vit_head_dim]),
                                     reshape_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     offset=[0, 0, 1, 0, 0],
                                     steps=[1, 1, 1, 1, 1],
                                     ends=[int64_max, int64_max, 2, int64_max, int64_max],
                                     hasparamConvert_axes=[2],
                                     loc=L(attn_qkv + ".k.slice"),
                                     ip=ip).output
            v_slice_op = top.SliceOp(T([1, 1025, 1, self.vit_num_heads, self.vit_head_dim]),
                                     reshape_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     offset=[0, 0, 2, 0, 0],
                                     steps=[1, 1, 1, 1, 1],
                                     ends=[int64_max, int64_max, 3, int64_max, int64_max],
                                     hasparamConvert_axes=[2],
                                     loc=L(attn_qkv + ".v.slice"),
                                     ip=ip).output
            q_squeeze_op = top.SqueezeOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         q_slice_op,
                                         axes=[2],
                                         loc=L(attn_qkv + ".q.squeeze"),
                                         ip=ip).output
            k_squeeze_op = top.SqueezeOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         k_slice_op,
                                         axes=[2],
                                         loc=L(attn_qkv + ".k.squeeze"),
                                         ip=ip).output
            v_squeeze_op = top.SqueezeOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         v_slice_op,
                                         axes=[2],
                                         loc=L(attn_qkv + ".v.squeeze"),
                                         ip=ip).output

            # Q向量处理 (缩放和转置)
            q_permute_op = top.PermuteOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                         q_squeeze_op,
                                         order=[0, 2, 1, 3],# 维度重排: [0, 2, 1, 3]
                                         loc=L(attn_qkv + ".q.permute"),
                                         ip=ip).output
            q_mulconst_op = top.MulConstOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                           q_permute_op,
                                           const_val=0.125,# 缩放因子(1/sqrt(d_k))
                                           loc=L(attn_qkv + ".q.mulconst"),
                                           ip=ip).output

            # K向量处理 (两次转置)
            k_permute_op0 = top.PermuteOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                          k_squeeze_op,
                                          order=[0, 2, 1, 3],
                                          loc=L(attn_qkv + ".k.permute0"),
                                          ip=ip).output
            k_permute_op1 = top.PermuteOp(T([1, self.vit_num_heads, self.vit_head_dim, 1025]),
                                          k_permute_op0,
                                          order=[0, 1, 3, 2],# 准备点积
                                          loc=L(attn_qkv + ".k.permute1"),
                                          ip=ip).output

            # QK点积 (注意力分数)
            qk_mm_op = top.MatMulOp(T([1, self.vit_num_heads, 1025, 1025]),
                                    q_mulconst_op,
                                    k_permute_op1,
                                    vit_mlir.none_op,
                                    loc=L(attn_qkv + ".qk.matmul"),
                                    ip=ip).output

            # Softmax归一化
            qk_softmax_op = top.SoftmaxOp(T([1, self.vit_num_heads, 1025, 1025]),
                                          qk_mm_op,
                                          axis=3,# 在最后一个维度softmax
                                          loc=L(attn_qkv + ".qk.softmax"),
                                          ip=ip).output

            # V向量转置
            v_permute_op = top.PermuteOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                         v_squeeze_op,
                                         order=[0, 2, 1, 3],
                                         loc=L(attn_qkv + ".v.permute"),
                                         ip=ip).output

            # 注意力加权求和
            qkv_mm_op = top.MatMulOp(T([1, self.vit_num_heads, 1025, self.vit_head_dim]),
                                     qk_softmax_op,
                                     v_permute_op,
                                     vit_mlir.none_op,
                                     loc=L(attn_qkv + ".qkv.matmul"),
                                     ip=ip).output

            # 输出恢复维度顺序
            o_permute_op = top.PermuteOp(T([1, 1025, self.vit_num_heads, self.vit_head_dim]),
                                         qkv_mm_op,
                                         order=[0, 2, 1, 3],
                                         loc=L(attn_proj + ".o.permute"),
                                         ip=ip).output

            # 拼接多头输出
            o_reshape_op = top.ReshapeOp(T([1, 1025, self.vit_hidden_size]),
                                         o_permute_op,
                                         shape=[1, 1025, self.vit_hidden_size],
                                         loc=L(attn_proj + ".o.reshape"),
                                         ip=ip).output

            # 输出投影层
            weight_op4 = vit_mlir.create_weight_op(attn_proj + ".weight",
                                                   [self.vit_hidden_size, self.vit_hidden_size])
            weight_op5 = vit_mlir.create_weight_op(attn_proj + ".bias",
                                                   [1, 1, self.vit_hidden_size])
            o_mm_op = top.MatMulOp(T([1, 1025, self.vit_hidden_size]),
                                   o_reshape_op,
                                   weight_op4,
                                   weight_op5,
                                   loc=L(attn_qkv + ".matmul"),
                                   ip=ip).output

            # 残差连接 (带缩放)
            weight_op6 = vit_mlir.create_weight_op(ls1, [1, 1, self.vit_hidden_size])
            o_mul_op = top.MulOp(T([1, 1025, self.vit_hidden_size]), [o_mm_op, weight_op6],
                                 loc=L(attn_proj + ".o.mul"),
                                 ip=ip).output
            new_op = top.AddOp(T([1, 1025, self.vit_hidden_size]), [in_op, o_mul_op],
                               loc=L(attn_proj + ".add"),
                               ip=ip).output
            return new_op

        # ----------------------- 前馈神经网络子层 -----------------------
        def vision_mlp(in_op):
            """构建FFN层的MLIR操作"""
            # 层归一化
            weight_op0 = vit_mlir.create_weight_op(norm2 + ".weight", [1, 1, self.vit_hidden_size])
            weight_op1 = vit_mlir.create_weight_op(norm2 + ".bias", [1, 1, self.vit_hidden_size])
            norm2_op = top.LayerNormOp(T([1, 1025, self.vit_hidden_size]),
                                       in_op,
                                       weight_op0,
                                       weight_op1,
                                       axis=2,
                                       eps=self.vit_ln_eps,
                                       normalized_shape=[self.vit_hidden_size],
                                       loc=L(norm2),
                                       ip=ip).output
            # 第一个全连接层 (扩展维度)
            weight_op2 = vit_mlir.create_weight_op(
                mlp + "1.weight", [self.vit_hidden_size, self.vit_intermediate_size])
            weight_op3 = vit_mlir.create_weight_op(mlp + "1.bias",
                                                   [1, 1, self.vit_intermediate_size])
            mlp_up_op = top.MatMulOp(T([1, 1025, self.vit_intermediate_size]),
                                     norm2_op,
                                     weight_op2,
                                     weight_op3,
                                     loc=L(mlp + "1"),
                                     ip=ip).output
            # 激活函数 (根据配置选择)
            active_op = self.activate(vit_mlir, mlp_up_op, self.vision_config.hidden_act, mlp)

            # 第二个全连接层 (恢复维度)
            weight_op4 = vit_mlir.create_weight_op(
                mlp + "2.weight", [self.vit_intermediate_size, self.vit_hidden_size])
            weight_op5 = vit_mlir.create_weight_op(mlp + "2.bias", [1, 1, self.vit_hidden_size])
            mlp_down_op = top.MatMulOp(T([1, 1025, self.vit_hidden_size]),
                                       active_op,
                                       weight_op4,
                                       weight_op5,
                                       loc=L(mlp + "2"),
                                       ip=ip).output

            # 残差连接 (带缩放)
            weight_op6 = vit_mlir.create_weight_op(ls2, [1, 1, self.vit_hidden_size])
            mul_op = top.MulOp(T([1, 1025, self.vit_hidden_size]), [mlp_down_op, weight_op6],
                               loc=L(mlp + ".mul"),
                               ip=ip).output
            new_op = top.AddOp(T([1, 1025, self.vit_hidden_size]), [in_op, mul_op],
                               loc=L(mlp + ".add"),
                               ip=ip).output
            return new_op
        # 顺序执行注意力和FFN子层
        in_op = vision_attention(in_op)
        in_op = vision_mlp(in_op)
        return in_op

    @override
    def gen_vit_mlir(self):
        """生成整个ViT的MLIR表示"""
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        # 权重文件路径
        vit_npz = "vit_top_weights.npz"
        patch_embed = "vision_model.embeddings"
        layers = "vision_model.encoder.layers"
        merger = "mlp1"# 视觉-语言融合模块

        # ----------------------- 权重保存函数 -----------------------
        def save_weights():
            """提取并保存ViT权重到NPZ文件"""
            weights_dict = {}

            # 嵌入层权重
            self.set_common_weight(patch_embed + ".class_embedding", weights_dict)
            self.set_common_weight(patch_embed + ".patch_embedding", weights_dict)
            self.set_common_weight(patch_embed + ".position_embedding", weights_dict)

            # 融合层权重
            self.set_common_weight(merger + ".0", weights_dict)
            self.set_linear_weight(merger + ".1", weights_dict)
            self.set_linear_weight(merger + ".3", weights_dict)

            # 各Transformer层权重
            for i in range(self.depth):
                # 层归一化权重
                self.set_common_weight(layers + f".{i}.norm1", weights_dict)
                self.set_common_weight(layers + f".{i}.norm2", weights_dict)
                # 缩放系数
                self.set_common_weight(layers + f".{i}.ls1", weights_dict)
                self.set_common_weight(layers + f".{i}.ls2", weights_dict)
                # 注意力权重
                self.set_linear_weight(layers + f".{i}.attn.proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.attn.qkv", weights_dict)
                # MLP权重
                self.set_linear_weight(layers + f".{i}.mlp.fc1", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.fc2", weights_dict)
            # save weights
            # 保存到NPZ文件
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        # ----------------------- MLIR生成主逻辑 -----------------------
        # 创建MLIR导入器 (输入: 图像张量, 输出: 图像token)
        vit_mlir = MLIRImporter([[1, 3, self.image_size, self.image_size]],
                                [[self.num_image_token, self.hidden_size]],
                                "vit",
                                Platform.LLM, ['F32'],
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point# 操作插入点

        # 工具函数
        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        # 输入节点 (图像像素值)
        in_op = vit_mlir.create_input_op(L('pixel_value'), 0)
        # embedding
        # ----------------------- Patch Embedding -----------------------
        # 卷积形式的分块嵌入 (核大小=分块大小, 步长=分块大小)
        weight_op0 = vit_mlir.create_weight_op(
            patch_embed + ".patch_embedding.weight",
            [self.vit_hidden_size, 3, self.patch_size, self.patch_size])
        weight_op1 = vit_mlir.create_weight_op(patch_embed + ".patch_embedding.bias",
                                               [self.vit_hidden_size])
        conv_op = top.ConvOp(T([1, self.vit_hidden_size, 32, 32]),# 输出特征图尺寸计算: 256/8=32
                             in_op,
                             weight_op0,
                             weight_op1,
                             kernel_shape=[self.patch_size, self.patch_size],
                             strides=[self.patch_size, self.patch_size],
                             pads=[0, 0, 0, 0],# 无填充
                             dilations=[1, 1],
                             loc=L(patch_embed),
                             ip=ip).output

        # 形状重塑: [B, C, H, W] -> [B, C, L] (L=H*W)
        reshape_op = top.ReshapeOp(T([1, 1024, self.vit_hidden_size]),
                                   conv_op,
                                   shape=[1, self.vit_hidden_size, -1],
                                   loc=L(patch_embed + ".reshape"),
                                   ip=ip).output

        # 维度转置: [B, C, L] -> [B, L, C]
        permute_op = top.PermuteOp(T([1, self.vit_hidden_size, 1024]),# 注意维度顺序变化
                                   reshape_op,
                                   order=[0, 2, 1],# 原[0,1,2] -> 新[0,2,1]
                                   loc=L(patch_embed + ".permute"),
                                   ip=ip).output

        # 添加[class] token
        weight_op2 = vit_mlir.create_weight_op(patch_embed + ".class_embedding",
                                               [1, 1, self.vit_hidden_size])
        concat_op = top.ConcatOp(T([1, 1025, self.vit_hidden_size]),# 1024+1=1025
                                 [weight_op2, permute_op],
                                 axis=1,# 在序列维度拼接
                                 loc=L(patch_embed + ".concat"),
                                 ip=ip).output

        # 添加位置编码
        weight_op3 = vit_mlir.create_weight_op(patch_embed + ".position_embedding",
                                               [1, 1025, self.vit_hidden_size])
        add_op = top.AddOp(T([1, 1025, self.vit_hidden_size]), [concat_op, weight_op3],
                           loc=L(patch_embed + ".add"),
                           ip=ip).output
        # block
        # ----------------------- Transformer 编码层 -----------------------
        new_op = add_op
        for idx in range(self.depth):
            new_op = self.vision_block(vit_mlir, idx, new_op)
        # merge
        # ----------------------- 视觉-语言融合模块 -----------------------
        # 移除[class] token (只保留图像token)
        slice_op = top.SliceOp(T([1, 1024, 1024]),
                               new_op,
                               vit_mlir.none_op,
                               vit_mlir.none_op,
                               vit_mlir.none_op,
                               offset=[0, 1, 0],# 跳过第一个token
                               steps=[1, 1, 1],
                               ends=[1, int64_max, 1024],
                               hasparamConvert_axes=[1],# 指定序列维度
                               loc=L(merger + ".slice"),
                               ip=ip).output

        # 空间结构重建 (1024个token -> 32x32特征图)
        reshape_op0 = top.ReshapeOp(T([1, 32, 16, 2048]),
                                    slice_op,
                                    shape=[1, 32, 16, 2048],
                                    loc=L(merger + ".reshape0"),
                                    ip=ip).output

        # 第一次空间下采样 (32x32 -> 32x16)
        permute_op0 = top.PermuteOp(T([1, 16, 32, 2048]),
                                    reshape_op0,
                                    order=[0, 2, 1, 3],
                                    loc=L(merger + ".permute0"),
                                    ip=ip).output

        # 第二次空间下采样 (32x16 -> 16x16)
        reshape_op1 = top.ReshapeOp(T([1, 16, 16, self.vit_intermediate_size]),
                                    permute_op0,
                                    shape=[1, 16, 16, self.vit_intermediate_size],
                                    loc=L(merger + ".reshape1"),
                                    ip=ip).output
        permute_op1 = top.PermuteOp(T([1, 16, 16, self.vit_intermediate_size]),
                                    reshape_op1,
                                    order=[0, 2, 1, 3],
                                    loc=L(merger + ".permute1"),
                                    ip=ip).output

        # 展平空间维度 (16x16=256个token)
        reshape_op2 = top.ReshapeOp(T([1, self.num_image_token, self.vit_intermediate_size]),
                                    permute_op1,
                                    shape=[1, -1, self.vit_intermediate_size],
                                    loc=L(merger + ".reshape2"),
                                    ip=ip).output

        # 层归一化
        weight_op4 = vit_mlir.create_weight_op(merger + ".0.weight",
                                               [1, 1, self.vit_intermediate_size])
        weight_op5 = vit_mlir.create_weight_op(merger + ".0.bias",
                                               [1, 1, self.vit_intermediate_size])
        norm_op = top.LayerNormOp(T([1, self.num_image_token, self.vit_intermediate_size]),
                                  reshape_op2,
                                  weight_op4,
                                  weight_op5,
                                  axis=2,
                                  eps=self.vit_ln_eps,
                                  normalized_shape=[self.vit_intermediate_size],
                                  loc=L(merger + ".norm"),
                                  ip=ip).output

        # 投影到语言模型空间
        weight_op6 = vit_mlir.create_weight_op(merger + ".1.weight",
                                               [self.vit_intermediate_size, self.hidden_size])
        weight_op7 = vit_mlir.create_weight_op(merger + ".1.bias", [1, 1, self.hidden_size])
        mm_op0 = top.MatMulOp(T([1, self.num_image_token, self.hidden_size]),
                              norm_op,
                              weight_op6,
                              weight_op7,
                              loc=L(merger + ".1"),
                              ip=ip).output

        # 激活函数 (GELU)
        active_op = self.activate(vit_mlir, mm_op0, ActType.GELU, merger)
        # 最终投影层
        weight_op8 = vit_mlir.create_weight_op(merger + ".3.weight",
                                               [self.hidden_size, self.hidden_size])
        weight_op9 = vit_mlir.create_weight_op(merger + ".3.bias", [1, 1, self.hidden_size])
        mm_op1 = top.MatMulOp(T([1, self.num_image_token, self.hidden_size]),
                              active_op,
                              weight_op8,
                              weight_op9,
                              loc=L(merger + ".3"),
                              ip=ip).output

        # 输出适配语言模型
        reshape_op3 = top.ReshapeOp(T([self.num_image_token, self.hidden_size]),
                                    mm_op1,
                                    shape=[self.num_image_token, self.hidden_size],
                                    loc=L(merger + ".reshape3"),
                                    ip=ip).output

        # 创建输出节点
        vit_mlir.create_return_op([reshape_op3])

        # 写入MLIR文件
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)

        # 保存权重文件
        save_weights()
