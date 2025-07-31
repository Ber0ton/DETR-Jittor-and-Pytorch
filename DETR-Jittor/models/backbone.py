from collections import OrderedDict
from typing import Dict, List

import jittor as jt
from jittor import nn
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

# -----------------------------------------------------------------------------
# Helper layers & utils
# -----------------------------------------------------------------------------
class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed (frozen) affine parameters & running statistics."""
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        # affine parameters (never updated during training)
        self.weight = nn.Parameter(jt.ones(num_features), requires_grad=False)
        self.bias   = nn.Parameter(jt.zeros(num_features), requires_grad=False)

        # running stats (treated as constant buffers)
        self.register_buffer("running_mean", jt.zeros(num_features))
        self.register_buffer("running_var",  jt.ones(num_features))
        self.eps = eps

    def execute(self, x):
        # Reshape for broadcasting
        w  = self.weight.view(1, -1, 1, 1)
        b  = self.bias.view(1,  -1, 1, 1)
        rm = self.running_mean.view(1, -1, 1, 1)
        rv = self.running_var.view(1,  -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias  = b - rm * scale
        return x * scale + bias

# -----------------------------------------------------------------------------
#   IntermediateLayerGetter (minimal Jittor version)
# -----------------------------------------------------------------------------
class IntermediateLayerGetter(nn.Module):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]):
        super().__init__()
        self.return_layers = return_layers.copy()

        layers = OrderedDict()
        picked = 0
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                picked += 1
                if picked == len(return_layers):
                    break        # <-- 在此处停止；跳过 avgpool 和 fc
        self.body = nn.Sequential(layers)

    def execute(self, x):
        out = {}
        for name, module in self.body.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

# -----------------------------------------------------------------------------
#  Backbone implementation (ResNet with frozen BatchNorm)
# -----------------------------------------------------------------------------

# Jittor provides ResNet variants under `jittor.models`
from jittor.models import resnet18, resnet34, resnet50, resnet101

# A mapping to conveniently get the constructor by string name ("resnet50", ...)
_JT_RESNETS = {
    "resnet18" : resnet18,
    "resnet34" : resnet34,
    "resnet50" : resnet50,
    "resnet101": resnet101,
}

class BackboneBase(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_layers: bool):
        super().__init__()

        # Freeze parameters (stop gradients) except for when training specified layers
        for name, param in backbone.named_parameters():
            if (not train_backbone) or ("layer2" not in name and "layer3" not in name and "layer4" not in name):
                # `stop_grad()` turns off gradient tracking in Jittor
                param.stop_grad()

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers)
        self.num_channels = num_channels

    def execute(self, tensor_list):
        # tensor_list is expected to be a util.misc.NestedTensor instance
        xs = self.body(tensor_list.tensors)
        out: Dict[str, "NestedTensor"] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None, "NestedTensor mask is required"
            mask = nn.interpolate(m.unsqueeze(0).float(), size=x.shape[-2:]).to(jt.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with **frozen BatchNorm** using Jittor."""
    def __init__(self,
                 name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if name not in _JT_RESNETS:
            raise ValueError(f"Unsupported ResNet variant '{name}' for Jittor backend.")

        # Build the ResNet; replace norm layers with FrozenBatchNorm2d
        def _replace_bn(module):
            for child_name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, child_name, FrozenBatchNorm2d(child.num_features))
                else:
                    _replace_bn(child)
            return module

        backbone = _JT_RESNETS[name](pretrained=is_main_process())
        backbone = _replace_bn(backbone)

        # Support dilation on layer4 (similar to PyTorch's replace_stride_with_dilation)
        if dilation:
            # Jittor's ResNet allows modifying the stride and dilation of layer4's first block
            backbone.layer4[0].conv1.stride = (1, 1)
            backbone.layer4[0].conv2.dilation = (2, 2)
            backbone.layer4[0].conv2.padding  = (2, 2)

        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

# -----------------------------------------------------------------------------
#  Joiner = Backbone + Positional Encoding wrapper
# -----------------------------------------------------------------------------
class Joiner(nn.Sequential):
    def __init__(self, backbone: BackboneBase, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def execute(self, tensor_list):
        xs = self[0](tensor_list)  # Backbone output (dict[str, NestedTensor])
        out: List["NestedTensor"] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # Positional encoding needs same dtype as the feature map
            pos.append(self[1](x).astype(x.tensors.dtype))
        return out, pos

# -----------------------------------------------------------------------------
#  Factory
# -----------------------------------------------------------------------------

def build_backbone(args):
    # build_position_encoding should already be ported to Jittor
    position_embedding = build_position_encoding(args)
    train_backbone     = args.lr_backbone > 0
    return_interm      = args.masks

    backbone = Backbone(args.backbone, train_backbone, return_interm, args.dilation)
    model    = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
