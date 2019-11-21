import torch
import torch.nn as nn

from ssd.detector import Detector
from utils.arguments import Arguments


def arguments(parser):
    parser.add_argument('--format', required=False, type=str, default='torch',
                        choices=[
                            "torch",
                            # "onnx",
                            # "tf",
                        ],
                        help="Export format")


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> nn.Module:

    model.load(torch.load(args.model, map_location=lambda s, l: s))
    model.eval()
    Detector.init(model.num_classes, args.batch, nms=False)

    return model


def export(model: nn.Module, args: Arguments.parse.Namespace = None, **kwargs):
    inputs = torch.FloatTensor(args.batch, 3, 300, 300)
    _ = model(inputs)

    module = torch.jit.trace(model, inputs)
    module.save(args.dest)
