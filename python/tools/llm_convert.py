#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import pymlir


def parse_max_pixels(value):
    """
    解析图像的最大像素数参数。

    功能：
    - 如果输入是单个数字（如 "100000"），则直接转换为整数返回。
    - 如果输入包含逗号（如 "128,124"），则解析为宽度和高度的整数，最后返回宽*高的像素数。

    参数：
        value (str): 用户输入的字符串，可能是单个数字或两个用逗号分隔的数字。

    返回：
        int: 最大像素数（直接的整数或宽×高计算后的结果）。

    异常：
        argparse.ArgumentTypeError: 当输入格式不符合要求时抛出异常。
    """

    if ',' in value:
        parts = value.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "The input must be two integers separated by a comma, e.g., 128,124")
        try:
            width = int(parts[0].strip())
            height = int(parts[1].strip())
        except ValueError:
            raise argparse.ArgumentTypeError("The input values must be integers, e.g., 128,124")
        return int(width * height)
    else:
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "The input must be an integer or two integers separated by a comma, e.g., 128,124")


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='original weight, like ./Qwen2-7B-Instruct')
    parser.add_argument('-s', '--seq_length', type=int, required=True,
                        help="sequence length")
    parser.add_argument('-q', '--quantize', type=str, required=True,
                        choices=["bf16", "w8bf16", "w4bf16", "f16", "w8f16", "w4f16"],
                        help="quantize type for bmodel")
    parser.add_argument('-g', "--q_group_size", default=64, type=int,
                        help="group size for per-group quant, only used in quant mode")
    parser.add_argument('-c', '--chip', type=str, default="bm1684x",
                        choices=["bm1684x", "bm1688", "cv186x"],
                        help="chip type for bmodel")
    parser.add_argument('--num_device', type=int, default=1,
                        help="num device for bmodel")
    parser.add_argument('--num_core', type=int, default=0, help="num cores for bmodel")
    parser.add_argument('--symmetric', action='store_true', help='do symmetric quantize')
    parser.add_argument('--embedding_disk', action='store_true',
                        help='export embedding as bin file and inference by cpu')
    parser.add_argument('--do_sample', action='store_true',
                        help='Add sample head and separate greedy head from lmhead')
    parser.add_argument('--use_block_with_kv', action='store_true',
                        help='use history kv for prefill, default is False')
    parser.add_argument('--max_input_length', type=int, default=0,
                        help='max input length for prefill, default 0 means the same as seq_length')
    parser.add_argument('--max_prefill_kv_length', type=int, default=0,
                        help='max prefill kv length, default 0 means the same as seq_length')
    parser.add_argument('--max_pixels', type=parse_max_pixels, default=0,
                        help="max pixels for vit, for example: 240,420 or 100800")
    parser.add_argument('--dynamic', action='store_true',
                        help='enable dynamic compiling for prefill, not recommended')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode, temp files will not be deleted')
    parser.add_argument("-V", "--version", action='version', version='%(prog)s ' + pymlir.__version__)
    parser.add_argument('-o', '--out_dir', type=str, default='./tmp',
                        help='output mlir/bmodel path, default `./tmp`')
    args = parser.parse_args()
    # yapf: enable
    # 如果启用了 use_block_with_kv 功能( prefill 时使用历史 kv)
    if args.use_block_with_kv:
        # 如果没有显式设置 max_input_length (<=0),则默认设置为 seq_length // 4
        if args.max_input_length <= 0:
            args.max_input_length = args.seq_length // 4
            print("Warning: max_input_length is not set, use seq_length // 4 as default value: {}".
                  format(args.max_input_length))
        # 如果用户设置的 max_input_length 过大（超过 seq_length // 2），则报错
        elif args.max_input_length > args.seq_length // 2:
            raise ValueError(
                "max_input_length should not be larger than seq_length // 2, got: {}".format(
                    args.max_input_length))
    # 如果 max_prefill_kv_length 没有显式设置 (<=0)，则默认使用 seq_length
    if args.max_prefill_kv_length <= 0:
        args.max_prefill_kv_length = args.seq_length
    # 如果设置的 max_prefill_kv_length 过大（超过 seq_length），则报错
    elif args.max_prefill_kv_length > args.seq_length:
        raise ValueError(
            "max_prefill_kv_length should not be larger than seq_length, got: {}".format(
                args.max_prefill_kv_length))

    # 根据模型类型选择不同的转换器
    from transformers import AutoConfig

    # 从指定模型路径加载配置文件
    # trust_remote_coda = True 允许加载模型时执行其自定义的 Python 代码
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if config.model_type in ["qwen3", "qwen2", "llama", "minicpm"]:
        # 针对 Qwen、LLaMA、MiniCPM 等模型的通用转换器
        from llm.LlmConverter import LlmConverter
        converter = LlmConverter(args, config)
    elif config.model_type in ["chatglm"]:
        # 针对 ChatGLM 模型的专用转换器
        from llm.Chatglm3Converter import Chatglm3Converter
        converter = Chatglm3Converter(args, config)
    elif config.model_type in ["phi3"]:
        # 针对 Phi3 模型的专用转换器
        from llm.Phi3Converter import Phi3Converter
        converter = Phi3Converter(args, config)
    elif config.model_type in ['qwen2_vl']:
        # 针对 Qwen2-VL（视觉+语言模型）的转换器
        from llm.Qwen2VLConverter import Qwen2VLConverter
        converter = Qwen2VLConverter(args, config)
    elif config.model_type in ['qwen2_5_vl']:
        # 针对 Qwen2.5-VL（视觉+语言模型）的转换器
        from llm.Qwen2_5VLConverter import Qwen2_5VLConverter
        converter = Qwen2_5VLConverter(args, config)
    elif config.model_type in ['internvl_chat']:
        # 针对 InternVL Chat 模型的转换器
        from llm.InternVL3Converter import InternVL3Converter
        converter = InternVL3Converter(args, config)
    elif config.model_type in ['gemma3']:
        # 针对 Gemma3 模型的转换器
        from llm.Gemma3Converter import Gemma3Converter
        converter = Gemma3Converter(args, config)
    else:
        # 如果模型类型不在上述支持列表中，则抛出异常
        raise RuntimeError("Unsupported model type: {}".format(config.model_type))
    converter.run()
