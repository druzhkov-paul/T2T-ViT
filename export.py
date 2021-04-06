import argparse
from functools import wraps

import numpy as np
import timm
import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op, get_registered_op, is_registered_op
from torch.onnx.symbolic_opset9 import _convolution
from timm.models import load_checkpoint
from onnx.optimizer import optimize

from models.t2t_vit import *
from utils import load_for_transfer_learning

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model_name', help='')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                        help='use ema version of weights if present')
    # parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    # parser.add_argument(
    #     '--shape',
    #     type=int,
    #     nargs='+',
    #     default=[1, 3, 256, 192],
    #     help='input size')
    args = parser.parse_args()
    return args

# from torch.onnx.symbolic_helper import parse_args

# def std_mean_symbolic(g, input, dim, unbiased=True, keepdim=False):
#     mean = g.op('ReduceMean', input, axes_i=dim, keepdims_i=int(keepdim))
#     std = g.op('Sqrt', g.op('ReduceSumSquare', input - mean, axes_i=dim, keepdims_i=int(keepdim)))
#     return mean, std


def optimize_onnx_graph(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)

    onnx_model = optimize(onnx_model, ['extract_constant_to_initializer',
                                       'eliminate_unused_initializer'])

    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(onnx_model, onnx_model_path)

def norm_symb(g, self, x):
    print('bbbbbbbbbbbbbbbbbbbbbb')
    return g.op('Norm', x)

def attn_symb(g, self, x):
    return g.op('Attn', x)

def mlp_symb(g, self, x):
    return g.op('Mlp', x)

if __name__ == '__main__':
    args = parse_args()

    # model = T2t_vit_14()
    # # load the preatrained weights
    # load_for_transfer_learning(model, 'data/models/71.7_T2T_ViT_7.pth.tar', use_ema=True, strict=False, num_classes=1000)  # change num_classes based on dataset

    # torch.onnx.symbolic_registry.register_op('norm1', norm_symb, 'timm_custom', args.opset_version)
    # torch.onnx.symbolic_registry.register_op('norm2', norm_symb, 'timm_custom', args.opset_version)
    # torch.onnx.symbolic_registry.register_op('attn', attn_symb, 'timm_custom', args.opset_version)
    # torch.onnx.symbolic_registry.register_op('mlp', mlp_symb, 'timm_custom', args.opset_version)

    model = timm.create_model(args.model_name, pretrained=args.pretrained, exportable=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    # model.load_state_dict(torch.load('model_best.pth.tar', map_location=device))
    model.eval()
    print(model.default_cfg)
    # input_shape = (1, ) + model.default_cfg['test_input_size']
    input_shape = (1, ) + model.default_cfg['input_size']
    print(input_shape)
    dummy_input = torch.randn(*input_shape)
    depth = model.depth
    torch.onnx.export(
        model,
        dummy_input,
        args.output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=False,
        opset_version=args.opset_version,
        input_names=['image'],
        # output_names=['t2t', 'pre_blk', 'feats'] + [f'blck_{i}' for i in range(depth)] + ['probs'],
        output_names=['probs'],
        strip_doc_string=True,
        enable_onnx_checker=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        dynamic_axes = {
            "image": {
                0: "batch_size",
                2: "height",
                3: "width"
            }
        }
    )
    optimize_onnx_graph(args.output_file)
