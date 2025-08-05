# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class Qwen2_5VLConverter(LlmConverter):
    """
    Qwen2.5 多模态（视觉-语言）模型转换器类
    继承自 LlmConverter，主要用于模型权重加载与位置编码处理。
    """
    def __init__(self, args, config):
        """
        初始化转换器，检查输入参数并提取视觉相关配置。
        :param args: 启动参数（包含 max_pixels 等参数）
        :param config: 模型配置（包含视觉分支配置 vision_config）
        """
        super().__init__(args, config)
        self.max_pixels = args.max_pixels
        # 图像分辨率检查
        if args.max_pixels == 0:
            raise RuntimeError("max_pixels is 0, please set max_pixels to a value greater than 0.")
        if args.max_pixels % (28 * 28) != 0:
            raise RuntimeError(
                "max_pixels is not a multiple of 28*28, please set max_pixels to a value that is a multiple of 28*28."
            )
        self.do_vit = True# 表示启用视觉Transformer分支
        # vision config

        # 提取视觉分支配置（通常来自预训练模型的 vision_config）
        self.vconfig = self.config.vision_config
        self.patch_size = self.vconfig.patch_size# 图像分块大小
        self.temporal_patch_size = self.vconfig.temporal_patch_size# 时序维度分块大小
        self.spatial_merge_size = self.vconfig.spatial_merge_size# 空间合并大小
        self.in_channels = self.vconfig.in_chans# 输入通道数 (如RGB=3)
        self.depth = self.vconfig.depth# 视觉Transformer层数

        # 计算图像patch数 = max_pixels / (patch_size^2)
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)

        # patch 向量维度 = 输入通道 * patch_size^2 * temporal_patch_size
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
        self.embed_dim = self.vconfig.hidden_size# patch嵌入后的维度
        self.vnum_heads = self.vconfig.num_heads# 视觉注意力头数
        self.vhead_dim = self.embed_dim // self.vnum_heads# 单个注意力头维度
        self.vintermediate_size = self.vconfig.intermediate_size# MLP中间层维度
        self.position_shape = [3, self.max_input_length]# 位置编码维度（3表示空间3维）
        self.fullatt_block_indexes = self.vconfig.fullatt_block_indexes# 全注意力层索引

    @override
    def load_pretrained(self, config):
        """
        加载预训练模型权重
        :param config: 模型配置
        """
        super().load_pretrained(config)
        self.llm_type = LlmType.QWEN2# 设置语言模型类型为 QWEN2

    @override
    def rotary_embedding(self):
        """
        生成 RoPE（旋转位置编码）cos/sin 参数
        用于语言模型注意力层
        """
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
        rotary_embed = Qwen2VLRotaryEmbedding(self.config)

        # 生成位置索引 [1, 1, seq_length] -> 扩展到 [3, 1, seq_length]
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(
            1, 1, self.seq_length).expand(3, 1, self.seq_length)

        # 输入dummy张量
        x = torch.zeros([1, self.seq_length, self.hidden_size], dtype=torch.float32)

        # 获取 cos/sin
        cos, sin = rotary_embed(x, position_ids)

        # reshape到 [seq_length, 1, head_dim]
        cos = cos[0].reshape(self.seq_length, 1, -1)
        sin = sin[0].reshape(self.seq_length, 1, -1)
        assert (cos.shape[-1] == self.head_dim)
        assert (sin.shape[-1] == self.head_dim)
        # half
        # 取一半维度 (RoPE 通常对偶数维度进行拆分)
        cos = cos[:, :, :self.head_dim // 2]
        sin = sin[:, :, :self.head_dim // 2]
        return cos.numpy(), sin.numpy()  #[seq, 1, 64] # 返回 numpy 数组，形状 [seq, 1, 64]

    def mrope(self, mlir_gen, in_op, name: str):
        """
        MRoPE (多维旋转位置编码) 的 MLIR 表达式生成
        通过 slice + concat + tile 组合构造多维RoPE权重。
        :param mlir_gen: MLIR 生成器
        :param in_op: 输入操作（包含位置信息）
        :param name: 权重名称
        """
        dim = in_op.type.shape[-1]
        # 生成权重Op
        weight_op = mlir_gen.create_weight_op(name + ".weight",
                                              [self.seq_length, 1, self.head_dim // 2])
        # 从输入中按轴0聚合
        in_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, self.head_dim // 2]),
                             weight_op,
                             in_op,
                             axis=0,
                             loc=self.get_loc(name, mlir_gen),
                             ip=mlir_gen.insert_point).output

        # MRoPE分段信息 (时间/高度/宽度)
        mrope_section = getattr(self.config.rope_scaling, 'mrope_section', [16, 24, 24])
        t_dim, h_dim, w_dim = mrope_section  # 16,24,24# 默认 = [16, 24, 24]
        # slice cos_op = [1, dim 1, t_dim] + [1, dim, 1, h_dim] + [1, dim, 1, w_dim]

        # slice: 时间维度
        t_op = top.SliceOp(mlir_gen.get_tensor_type([1, dim, 1, t_dim]),
                           in_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           offset=[0, 0, 0, 0],
                           steps=[1, 1, 1, 1],
                           ends=[1, dim, 1, t_dim],
                           loc=self.get_loc(name + ".slice.t", mlir_gen),
                           ip=mlir_gen.insert_point).output

        # slice: 高度维度
        h_op = top.SliceOp(mlir_gen.get_tensor_type([1, dim, 1, h_dim]),
                           in_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           offset=[1, 0, 0, t_dim],
                           steps=[1, 1, 1, 1],
                           ends=[2, dim, 1, t_dim + h_dim],
                           loc=self.get_loc(name + ".slice.h", mlir_gen),
                           ip=mlir_gen.insert_point).output

        # slice: 宽度维度
        w_op = top.SliceOp(mlir_gen.get_tensor_type([1, dim, 1, w_dim]),
                           in_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           offset=[2, 0, 0, t_dim + h_dim],
                           steps=[1, 1, 1, 1],
                           ends=[3, dim, 1, t_dim + h_dim + w_dim],
                           loc=self.get_loc(name + ".slice.w", mlir_gen),
                           ip=mlir_gen.insert_point).output

        # concat: 合并 t/h/w 三段
        concat_op = top.ConcatOp(mlir_gen.get_tensor_type([1, dim, 1, t_dim + h_dim + w_dim]),
                                 [t_op, h_op, w_op],
                                 axis=3,
                                 loc=self.get_loc(name + ".concat", mlir_gen),
                                 ip=mlir_gen.insert_point).output

        # tile: 扩展两倍得到最终 head_dim
        tile_op = top.TileOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                             concat_op,
                             tile=[1, 1, 1, 2],
                             loc=self.get_loc(name + ".tile", mlir_gen),
                             ip=mlir_gen.insert_point).output
        return tile_op

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        """
        应用旋转位置编码 (RoPE/MRoPE) 到 q/k 投影
        :param mlir_gen: MLIR 生成器
        :param pos_op: 位置输入
        :param q_op: q投影操作
        :param k_op: k投影操作
        :param rotary_cos: cos参数名
        :param rotary_sin: sin参数名
        """
        # 生成 cos / sin MRoPE
        # cos MROPE
        cos_op = self.mrope(mlir_gen, pos_op, rotary_cos)
        # sin MROPE
        sin_op = self.mrope(mlir_gen, pos_op, rotary_sin)
        # ===== q_proj rotary ========
        # 应用到q/k
        q_op = self.rotary_pos(mlir_gen, q_op, cos_op, sin_op, "q_proj")

        # ===== k_proj rotary ========
        k_op = self.rotary_pos(mlir_gen, k_op, cos_op, sin_op, "k_cache")
        return q_op, k_op

    def vision_rotary(self):
        """
        视觉分支使用的旋转位置编码 (RoPE) 生成
        """
        from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding
        head_dim = self.vconfig.hidden_size // self.vnum_heads
        rotary_embed = VisionRotaryEmbedding(head_dim // 2)
        freqs = rotary_embed(self.num_patches)
        return freqs.cos().numpy(), freqs.sin().numpy()

    def vision_block(self, vit_mlir, id: int, in_op, cos_op, sin_op, mask_op):
        """构建单个视觉Transformer模块"""
        # 定义当前模块的权重路径
        norm1 = f"visual.blocks.{id}.norm1"# 第一个RMSNorm层
        attn_q = f"visual.blocks.{id}.attn.q"# 查询权重
        attn_k = f"visual.blocks.{id}.attn.k"# 键权重
        attn_v = f"visual.blocks.{id}.attn.v"# 值权重
        attn_proj = f"visual.blocks.{id}.attn.proj"# 注意力输出投影
        norm2 = f"visual.blocks.{id}.norm2"# 第二个RMSNorm层
        ip = vit_mlir.insert_point# MLIR操作插入点

        # 工具函数简化代码
        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)# 获取张量类型

        def L(name: str):
            return self.get_loc(name, vit_mlir)# 获取位置标记

        # ----------------------- 自注意力子层 -----------------------
        def vision_attention(in_op):
            """构建注意力机制的MLIR操作"""
            # 输入归一化 (RMSNorm)
            norm1_op = self.rms_norm(vit_mlir, in_op, norm1)

            # 计算Q/K/V投影
            hidden_shape = [self.num_patches, self.embed_dim]
            q_op = self.linear(vit_mlir,
                               attn_q,
                               norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            k_op = self.linear(vit_mlir,
                               attn_k,
                               norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            v_op = self.linear(vit_mlir,
                               attn_v,
                               norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)

            # 重塑为多头格式 [num_patches, embed_dim] -> [1, num_patches, num_heads, head_dim]
            qk_shape = [1, self.num_patches, self.vnum_heads, self.vhead_dim]
            q_op = top.ReshapeOp(T(qk_shape), q_op, loc=L(attn_q + ".reshape"), ip=ip).output

            k_op = top.ReshapeOp(T(qk_shape), k_op, loc=L(attn_k + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(qk_shape), v_op, loc=L(attn_v + ".reshape"), ip=ip).output

            # 应用旋转位置编码 (RoPE)
            q_op = self.rotary_pos(vit_mlir, q_op, cos_op, sin_op, attn_q + ".rotary")
            k_op = self.rotary_pos(vit_mlir, k_op, cos_op, sin_op, attn_k + ".rotary")

            # 使用FlashAttention计算注意力
            fa_op = top.FAttentionOp(T(qk_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     mask_op,
                                     vit_mlir.none_op,
                                     scale=self.vhead_dim**-0.5,# 缩放因子
                                     batch=1,
                                     q_head=self.vnum_heads,
                                     kv_head=self.vnum_heads,
                                     dim=self.vhead_dim,
                                     mq=self.num_patches,
                                     mk=self.num_patches,
                                     loc=L(f"visual.blocks.{id}.fattention"),
                                     ip=ip).output

            # 重塑回原始形状 [1, num_patches, num_heads, head_dim] -> [num_patches, embed_dim]
            fa_op = top.ReshapeOp(T(hidden_shape),
                                  fa_op,
                                  loc=L(f"visual.blocks.{id}.fattention.reshape"),
                                  ip=ip).output

            # 输出投影层
            out_op = self.linear(vit_mlir,
                                 attn_proj,
                                 fa_op, [self.embed_dim, self.embed_dim],
                                 [self.num_patches, self.embed_dim],
                                 force_bias=True)

            # 残差连接
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + ".add"),
                               ip=ip).output
            return out_op

        # ----------------------- 前馈神经网络子层 -----------------------
        def vision_mlp(in_op):
            """构建FFN层的MLIR操作"""
            in_shape = [self.num_patches, self.embed_dim]
            # 定义MLP权重路径
            mlp_gate = f"visual.blocks.{id}.mlp.gate_proj"  # 门控投影
            mlp_up = f"visual.blocks.{id}.mlp.up_proj"      # 上投影
            mlp_down = f"visual.blocks.{id}.mlp.down_proj"  # 下投影

            # 输入归一化 (RMSNorm)
            new_op = self.rms_norm(vit_mlir, in_op, norm2)

            # 门控激活分支
            gate_op = self.linear(vit_mlir, mlp_gate, new_op,
                                  [self.embed_dim, self.vintermediate_size],
                                  [self.num_patches, self.vintermediate_size])
            act_op = self.activate(vit_mlir, gate_op, self.vconfig.hidden_act, mlp_gate)

            # 上投影分支
            up_op = self.linear(vit_mlir, mlp_up, new_op, [self.embed_dim, self.vintermediate_size],
                                [self.num_patches, self.vintermediate_size])

            # 门控乘法 (Swish激活函数)
            new_op = top.MulOp(T([self.num_patches, self.vintermediate_size]), [act_op, up_op],
                               loc=L(mlp_up + ".mul"),
                               ip=ip).output

            # 下投影层
            down_op = self.linear(vit_mlir, mlp_down, new_op,
                                  [self.vintermediate_size, self.embed_dim], in_shape)

            # 残差连接
            new_op = top.AddOp(T(in_shape), [in_op, down_op], loc=L(mlp_down + ".add"),
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
        patch_embed = "visual.patch_embed.proj"# 图像分块嵌入
        rotary_cos = "visual.rotary.cos" # 旋转位置编码余弦分量
        rotary_sin = "visual.rotary.sin"# 旋转位置编码正弦分量
        merger_ln_q = "visual.merger.ln_q"# 融合层归一化
        merger_mlp0 = "visual.merger.mlp.0"# 融合层MLP第一部分
        merger_mlp2 = "visual.merger.mlp.2"# 融合层MLP第三部分

        # ----------------------- 权重保存函数 -----------------------
        def save_weights():
            """提取并保存ViT权重到NPZ文件"""
            # 生成旋转位置编码
            cos, sin = self.vision_rotary()
            weights_dict = {
                rotary_cos + ".weight": cos,
                rotary_sin + ".weight": sin,
            }

            # 处理图像分块嵌入权重
            data = self.model.read(patch_embed + ".weight").reshape(self.embed_dim, self.patch_dim)
            data = np.ascontiguousarray(np.transpose(data, (1, 0)))# 转置为[输入维度, 输出维度]
            weights_dict[patch_embed + ".weight"] = data

            # 融合层权重
            self.set_common_weight(merger_ln_q, weights_dict)
            self.set_linear_weight(merger_mlp0, weights_dict)
            self.set_linear_weight(merger_mlp2, weights_dict)

            # 各Transformer层权重
            for i in range(self.depth):
                # 层归一化权重
                self.set_common_weight(f"visual.blocks.{i}.norm1", weights_dict)
                self.set_common_weight(f"visual.blocks.{i}.norm2", weights_dict)

                # 注意力输出投影
                self.set_linear_weight(f"visual.blocks.{i}.attn.proj", weights_dict)

                # MLP权重
                self.set_linear_weight(f"visual.blocks.{i}.mlp.gate_proj", weights_dict)
                self.set_linear_weight(f"visual.blocks.{i}.mlp.up_proj", weights_dict)
                self.set_linear_weight(f"visual.blocks.{i}.mlp.down_proj", weights_dict)
                # split qkv
                # self.set_linear_weight(f"visual.blocks.{i}.attn.qkv", weights_dict)

                # 处理QKV权重 (拆分并转置)
                weight = self.model.read(f"visual.blocks.{i}.attn.qkv.weight").reshape(
                    3 * self.embed_dim, self.embed_dim)
                bias = self.model.read(f"visual.blocks.{i}.attn.qkv.bias").reshape(3 *
                                                                                   self.embed_dim)

                # 拆分Q/K/V权重
                q_w = weight[:self.embed_dim, :]
                k_w = weight[self.embed_dim:2 * self.embed_dim, :]
                v_w = weight[2 * self.embed_dim:, :]

                # 拆分Q/K/V偏置
                q_b = bias[:self.embed_dim]
                k_b = bias[self.embed_dim:2 * self.embed_dim]
                v_b = bias[2 * self.embed_dim:]

                # 转置权重矩阵 (适配MLIR格式)
                q_w = np.ascontiguousarray(np.transpose(q_w, (1, 0)))
                k_w = np.ascontiguousarray(np.transpose(k_w, (1, 0)))
                v_w = np.ascontiguousarray(np.transpose(v_w, (1, 0)))

                # 添加到权重字典
                weights_dict[f"visual.blocks.{i}.attn.q.weight"] = q_w
                weights_dict[f"visual.blocks.{i}.attn.k.weight"] = k_w
                weights_dict[f"visual.blocks.{i}.attn.v.weight"] = v_w
                weights_dict[f"visual.blocks.{i}.attn.q.bias"] = q_b
                weights_dict[f"visual.blocks.{i}.attn.k.bias"] = k_b
                weights_dict[f"visual.blocks.{i}.attn.v.bias"] = v_b
            # save weights
            # 保存到NPZ文件
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        # ----------------------- MLIR生成主逻辑 -----------------------
        # 输入形状定义
        in_shape = [self.num_patches, self.patch_dim]# 输入特征
        position_shape = [self.num_patches, 2]# 位置ID
        mask_shape = [1, 1, self.num_patches, self.num_patches]# 注意力掩码
        out_dim = self.num_patches // (self.spatial_merge_size**2)# 输出token数量
        out_shape = [out_dim, self.hidden_size] # 输出形状

        # 创建MLIR导入器 (多输入支持)
        input_shapes = [in_shape, position_shape, mask_shape, mask_shape, [out_dim]]
        input_types = ['F32', 'INT32', 'F32', 'F32', 'INT32']# 各输入数据类型

        vit_mlir = MLIRImporter(input_shapes, [out_shape],
                                "vit",
                                Platform.LLM,
                                input_types,
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point# 操作插入点

        # 工具函数
        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        # 创建输入节点
        in0_op = vit_mlir.create_input_op(L('input_states'), 0)# 输入特征
        in1_op = vit_mlir.create_input_op(L('position_ids'), 1)# 位置ID
        in2_op = vit_mlir.create_input_op(L('full_attn_mask'), 2)# 全注意力掩码
        in3_op = vit_mlir.create_input_op(L('window_attn_mask'), 3)# 窗口注意力掩码
        in4_op = vit_mlir.create_input_op(L('reverse_index'), 4)# 反转索引

        # ----------------------- Patch Embedding -----------------------
        # 图像分块嵌入 (矩阵乘法实现)
        new_weight = vit_mlir.create_weight_op(patch_embed + ".weight",
                                               [self.patch_dim, self.embed_dim])
        new_op = top.MatMulOp(T([self.num_patches, self.embed_dim]),
                              in0_op,
                              new_weight,
                              vit_mlir.none_op,
                              loc=L(patch_embed),
                              ip=ip).output

        # ----------------------- 旋转位置编码准备 -----------------------
        # 余弦分量处理
        new_weight = vit_mlir.create_weight_op(rotary_cos + ".weight", [self.num_patches, 20])
        cos_op = top.GatherOp(T([self.num_patches, 2, 20]),# 根据位置ID获取对应余弦值
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_cos),
                              ip=ip).output
        cos_op = top.ReshapeOp(T([1, self.num_patches, 1, 40]), # 重塑为多头格式
                               cos_op,
                               loc=L(rotary_cos + ".reshape"),
                               ip=ip).output
        cos_op = top.TileOp(T([1, self.num_patches, 1, self.vhead_dim]),# 平铺到每个注意力头
                            cos_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_cos + ".tile"),
                            ip=ip).output

        # 正弦分量处理 (同上)
        new_weight = vit_mlir.create_weight_op(rotary_sin + ".weight", [self.num_patches, 20])
        sin_op = top.GatherOp(T([self.num_patches, 2, 20]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_sin),
                              ip=ip).output
        sin_op = top.ReshapeOp(T([1, self.num_patches, 1, 40]),
                               sin_op,
                               loc=L(rotary_sin + ".reshape"),
                               ip=ip).output
        sin_op = top.TileOp(T([1, self.num_patches, 1, self.vhead_dim]),
                            sin_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_sin + ".tile"),
                            ip=ip).output

        # ----------------------- Transformer 编码层 -----------------------
        for id in range(self.depth):
            # 根据层索引选择注意力掩码 (全注意力或窗口注意力)
            mask_op = in2_op
            if id not in self.fullatt_block_indexes:
                mask_op = in3_op
            # 构建当前视觉块
            new_op = self.vision_block(vit_mlir, id, new_op, cos_op, sin_op, mask_op)

        # merge
        # ----------------------- 视觉-语言融合模块 -----------------------
        # 空间下采样前的归一化
        new_op = self.rms_norm(vit_mlir, new_op, merger_ln_q)

        # 空间下采样 (合并相邻特征)
        out_dim = self.embed_dim * (self.spatial_merge_size**2)# 下采样后特征维度
        in_dim = self.num_patches // (self.spatial_merge_size**2)# 下采样后token数量
        new_op = top.ReshapeOp(T([in_dim, out_dim]), new_op, loc=L(merger_ln_q + ".reshape"),
                               ip=ip).output

        # MLP融合层 (两层投影)
        new_op = self.linear(vit_mlir, merger_mlp0, new_op, [out_dim, out_dim], [in_dim, out_dim])
        new_op = self.activate(vit_mlir, new_op, ActType.GELU, merger_mlp0)
        new_op = self.linear(vit_mlir, merger_mlp2, new_op, [out_dim, self.hidden_size],
                             [in_dim, self.hidden_size])
        # reverse
        # 反转索引 (恢复原始空间顺序)
        new_op = top.GatherOp(T([in_dim, self.hidden_size]),
                              new_op,
                              in4_op,
                              axis=0,
                              loc=L(merger_mlp2 + ".reverse"),
                              ip=ip).output

        # 创建输出节点并保存MLIR文件
        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
        # 保存权重文件
        save_weights()
