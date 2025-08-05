# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# TODO: in Gemma3, the rms_norm weights should be f32

from .LlmConverter import *
from typing_extensions import override


class Gemma3Converter(LlmConverter):
    """Gemma3 模型转换器，继承自 LlmConverter"""
    def __init__(self, args, config):
        """初始化转换器配置"""
        super().__init__(args, config)
        self.do_vit = True# 启用视觉转换模块
        # ViT 输出强制转换为 BF16 格式（原始为 F16）
        self.vit_f16_out_bf16 = True  # Gemma3 vit is f16, but we force output to bf16

    # 加载预训练模型信息
    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = GEMMA3_INFO# 模型信息
        self.llm_config = config.text_config# 文本模型配置
        self.llm_type = LlmType.GEMMA3# 指定模型类型为 Gemma3

    # 初始化转换器配置
    @override
    def init_config(self):
        super().init_config()
        self.tie_word_embeddings = True
        self.do_lmhead_merge = self.tie_word_embeddings and not self.embedding_disk and self.num_device < 2

    # 生成视觉转换器(ViT)的MLIR
    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        vconfig = self.config.vision_config# 视觉模型配置
        image_size = vconfig.image_size# 输入图像尺寸
        patch_size = vconfig.patch_size# 图像分块大小
        patches_per_image = image_size // patch_size# 每行/列的块数
        num_patches = patches_per_image**2# 总块数
        embed_dim = vconfig.hidden_size# 嵌入维度
        mm_tokens_per_image = self.config.mm_tokens_per_image# 多模态token数量
        hidden_act = vconfig.hidden_act# 激活函数类型

        # 创建权重文件
        vit_npz = "vit_top_weights.npz"# 权重保存路径
        top_path = "vision_tower.vision_model"# 模型顶层路径
        # 定义各层路径
        post_layernorm = f"{top_path}.post_layernorm"
        patch_embedding = f"{top_path}.embeddings.patch_embedding"
        positional_embedding = f"{top_path}.embeddings.position_embedding"
        mm_projector_norm = f"multi_modal_projector.mm_soft_emb_norm"
        mm_projector_mm = f"multi_modal_projector.mm_input_projection_weight"

        # Placeholder for the actual implementation of generating MLIR for Siglip Vision
        # 保存权重到NPZ文件
        def save_weights():
            weights_dict = {}
            # 添加公共权重（层归一化）
            self.set_common_weight(post_layernorm, weights_dict)
            # 遍历所有Transformer层
            for idx in range(vconfig.num_hidden_layers):
                layer_path = f"{top_path}.encoder.layers.{idx}"
                # 添加层归一化权重
                self.set_common_weight(f"{layer_path}.layer_norm1", weights_dict)
                self.set_common_weight(f"{layer_path}.layer_norm2", weights_dict)
                # 添加线性层权重
                self.set_linear_weight(f"{layer_path}.mlp.fc1", weights_dict)
                self.set_linear_weight(f"{layer_path}.mlp.fc2", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.q_proj", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.k_proj", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.v_proj", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.out_proj", weights_dict)

            # refine position embedding
            # 处理位置嵌入
            pos_embed = self.model.read(positional_embedding + ".weight")
            pos_ids = np.arange(num_patches, dtype=np.int32).reshape((1, num_patches))
            pos_embed_data = pos_embed[pos_ids]
            weights_dict[positional_embedding + ".weight"] = pos_embed_data
            # refine patch embedding
            # 处理块嵌入
            patch_embed = self.model.read(patch_embedding + ".weight")
            # 重塑并转置权重 [C, H, W] -> [H*W, C]
            patch_embed_data = patch_embed.reshape(
                (embed_dim, -1)).transpose(1, 0)  #[3*14*14, embed_dim]

            weights_dict[patch_embedding + ".weight"] = patch_embed_data
            patch_embed_bias = self.model.read(patch_embedding + ".bias")
            weights_dict[patch_embedding + ".bias"] = patch_embed_bias
            # mm projector
            # 多模态投影器权重
            weights_dict[mm_projector_mm] = self.model.read(mm_projector_mm)

            # 添加RMS归一化权重
            self.set_common_weight(mm_projector_norm, weights_dict, WeightType.RMS_NORM)
            # save to npz
            # 保存为NPZ文件
            np.savez(vit_npz, **weights_dict)

        save_weights()# 执行权重保存
        # generate mlir
        # === 生成MLIR计算图 ===
        in_shape = [1, 3, image_size, image_size]# 输入形状 [batch, channel, height, width]
        out_shape = [1, mm_tokens_per_image, self.hidden_size]# 输出形状
        hidden_shape = [1, num_patches, embed_dim]# 中间表示形状

        # 创建MLIR导入器
        vit_mlir = MLIRImporter([in_shape], [out_shape],
                                "vit",
                                Platform.LLM, ["F32"],
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point# 获取当前插入点

        # 辅助函数：获取张量类型
        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        # 辅助函数：获取位置信息（用于调试）
        def L(name: str):
            return self.get_loc(name, vit_mlir)

        # === 输入预处理 ===
        in_op = vit_mlir.create_input_op(L('pixel_values'), 0)# 创建输入操作

        # 将图像分块处理： [1, 3, 224, 224] -> [1, 3, 16, 14, 16, 14]
        new_op = top.ReshapeOp(T(
            [1, 3, patches_per_image, patch_size, patches_per_image, patch_size]),
                               in_op,
                               loc=L("pixel_reshape"),
                               ip=ip).output

        # 维度重排： [1, 3, 16, 14, 16, 14] -> [1, 16, 16, 3, 14, 14]
        new_op = top.PermuteOp(T(
            [1, patches_per_image, patches_per_image, 3, patch_size, patch_size]),
                               new_op,
                               order=[0, 2, 4, 1, 3, 5],# 新维度顺序
                               loc=L("pixel_transpose"),
                               ip=ip).output

        # 合并块： [1, 16, 16, 3, 14, 14] -> [1, 256, 3*14*14]
        new_op = top.ReshapeOp(T([1, num_patches, 3 * patch_size * patch_size]),
                               new_op,
                               loc=L("pixel_reshape2"),
                               ip=ip).output

        # === 块嵌入投影 ===
        new_weight = vit_mlir.create_weight_op(patch_embedding + ".weight",
                                               [3 * patch_size * patch_size, embed_dim])


        new_bias = vit_mlir.create_weight_op(patch_embedding + ".bias", [1, 1, embed_dim])
        # 线性变换: [1, 256, 588] * [588, 1024] -> [1, 256, 1024]
        new_op = top.MatMulOp(T(hidden_shape),
                              new_op,
                              new_weight,
                              new_bias,
                              loc=L(patch_embedding),
                              ip=ip).output

        # === 位置嵌入 ===
        new_weight = vit_mlir.create_weight_op(positional_embedding + ".weight", hidden_shape)
        new_op = top.AddOp(T(hidden_shape), [new_op, new_weight],
                           loc=L(patch_embedding + ".add"),
                           ip=ip).output

        # === ViT层定义 ===
        # 定义视觉MLP模块
        def vision_mlp(in_op, layer_path):
            intermediate_size = vconfig.intermediate_size
            in_shape = [1, num_patches, embed_dim]
            # 层归一化
            new_op = self.layer_norm(vit_mlir,
                                     in_op,
                                     f"{layer_path}.layer_norm2",
                                     eps=vconfig.layer_norm_eps)
            # 线性层1 + 激活函数
            fc1_op = self.linear(vit_mlir, f"{layer_path}.mlp.fc1", new_op,
                                 [embed_dim, intermediate_size],
                                 [1, num_patches, intermediate_size])
            act_op = self.activate(vit_mlir, fc1_op, hidden_act, layer_path)

            # 线性层2 + 残差连接
            fc2_op = self.linear(vit_mlir, f"{layer_path}.mlp.fc2", act_op,
                                 [intermediate_size, embed_dim], in_shape)
            new_op = top.AddOp(T([1, num_patches, embed_dim]), [in_op, fc2_op],
                               loc=L(layer_path + ".add"),
                               ip=ip).output
            return new_op

        # 处理所有Transformer层
        for idx in range(vconfig.num_hidden_layers):
            layer_path = f"{top_path}.encoder.layers.{idx}"
            norm_path = f"{layer_path}.layer_norm1"
            residual_op = new_op# 保存残差连接点

            # 层归一化
            new_op = self.layer_norm(vit_mlir, new_op, norm_path, eps=vconfig.layer_norm_eps)

            # 自注意力机制
            # 计算Q/K/V投影
            q_op = self.linear(vit_mlir, f"{layer_path}.self_attn.q_proj", new_op,
                               [embed_dim, embed_dim], hidden_shape)
            k_op = self.linear(vit_mlir, f"{layer_path}.self_attn.k_proj", new_op,
                               [embed_dim, embed_dim], [1, num_patches, embed_dim])
            v_op = self.linear(vit_mlir, f"{layer_path}.self_attn.v_proj", new_op,
                               [embed_dim, embed_dim], [1, num_patches, embed_dim])

            # 重塑为多头格式
            head_dim = vconfig.hidden_size // vconfig.num_attention_heads
            new_shape = [1, num_patches, vconfig.num_attention_heads, head_dim]
            q_op = top.ReshapeOp(T(new_shape),
                                 q_op,
                                 loc=L(f"{layer_path}.self_attn.q_reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T(new_shape),
                                 k_op,
                                 loc=L(f"{layer_path}.self_attn.k_reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(new_shape),
                                 v_op,
                                 loc=L(f"{layer_path}.self_attn.v_reshape"),
                                 ip=ip).output

            # 注意力计算
            fa_op = top.FAttentionOp(T(hidden_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     scale=head_dim**-0.5,# 缩放因子
                                     batch=1,
                                     q_head=vconfig.num_attention_heads,
                                     kv_head=vconfig.num_attention_heads,
                                     dim=head_dim,
                                     mq=num_patches,
                                     mk=num_patches,
                                     loc=L(f"{layer_path}.fattention"),
                                     ip=ip).output

            # 输出投影 + 残差连接
            o_op = self.linear(vit_mlir, f"{layer_path}.self_attn.out_proj", fa_op,
                               [embed_dim, embed_dim], hidden_shape)
            new_op = top.AddOp(T(hidden_shape), [residual_op, o_op],
                               loc=L(f"{layer_path}.residual_add"),
                               ip=ip).output

            # MLP处理
            new_op = vision_mlp(new_op, layer_path)

        # === 后处理 ===
        # 最终层归一化
        new_op = self.layer_norm(vit_mlir, new_op, post_layernorm, eps=vconfig.layer_norm_eps)
        ## mm_projector
        # === 多模态投影器 ===
        # 调整维度: [1, 256, 1024] -> [1, 1024, 256]
        new_op = top.PermuteOp(T([1, embed_dim, num_patches]),
                               new_op,
                               order=[0, 2, 1],
                               loc=L("mm_projector_transpose"),
                               ip=ip).output

        # 重塑为空间格式: [1, 1024, 16, 16]
        new_op = top.ReshapeOp(T([1, embed_dim, patches_per_image, patches_per_image]),
                               new_op,
                               loc=L("mm_projector_reshape"),
                               ip=ip).output

        # 空间下采样 (平均池化)
        tokens_per_side = int(mm_tokens_per_image**0.5)# 例如: 256 -> 16
        kernel_size = patches_per_image // tokens_per_side# 池化核大小
        new_op = top.AvgPoolOp(T([1, embed_dim, tokens_per_side, tokens_per_side]),
                               new_op,
                               kernel_shape=[kernel_size, kernel_size],
                               strides=[kernel_size, kernel_size],
                               pads=[0, 0, 0, 0, 0, 0, 0, 0],# 无填充
                               loc=L("mm_projector_avgpool"),
                               ip=ip).output

        # 重塑: [1, 1024, 16, 16] -> [1, 1024, 256]
        new_op = top.ReshapeOp(T([1, embed_dim, mm_tokens_per_image]),
                               new_op,
                               loc=L("mm_projector_reshape2"),
                               ip=ip).output

        # 维度调整: [1, 1024, 256] -> [1, 256, 1024]
        new_op = top.PermuteOp(T([1, mm_tokens_per_image, embed_dim]),
                               new_op,
                               order=[0, 2, 1],
                               loc=L("mm_projector_transpose2"),
                               ip=ip).output

        # RMS归一化
        new_op = self.rms_norm(vit_mlir, new_op, mm_projector_norm, eps=vconfig.layer_norm_eps)

        # 线性投影到文本空间
        new_weight = vit_mlir.create_weight_op(mm_projector_mm, [embed_dim, self.hidden_size])
        new_op = top.MatMulOp(T([1, mm_tokens_per_image, self.hidden_size]),
                              new_op,
                              new_weight,
                              vit_mlir.none_op,# 无偏置
                              loc=L("mm_projector_matmul"),
                              ip=ip).output

        # === 输出处理 ===
        vit_mlir.create_return_op([new_op])# 设置返回操作
        mlir_txt = vit_mlir.print_module()# 生成MLIR文本

        # 保存到文件
        with open("vit.mlir", "w") as f:
            f.write(mlir_txt)
