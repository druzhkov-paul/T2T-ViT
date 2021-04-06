import argparse

import numpy as np
import timm
import torch
from onnx.optimizer import optimize
from timm.models import load_checkpoint

from models.t2t_vit import *

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model_name', help='')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                        help='use ema version of weights if present')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    args = parser.parse_args()
    return args


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

if __name__ == '__main__':
    args = parse_args()

    model = timm.create_model(args.model_name, pretrained=args.pretrained, exportable=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    model.eval()
    print(model.default_cfg)
    try:
        input_shape = (1, ) + model.default_cfg['test_input_size']
    except KeyError:
        input_shape = (1, ) + model.default_cfg['input_size']
    print(input_shape)
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        args.output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=False,
        opset_version=args.opset_version,
        input_names=['image'],
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
