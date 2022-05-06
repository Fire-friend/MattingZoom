from .wrapper import *

# ------------------------------------------------------------------------------
#  Replaceable Backbones
# ------------------------------------------------------------------------------

SUPPORTED_BACKBONES = {
    'mobilenetv2': MobileNetV2Backbone,
    'mobilenetv2_human': MobileNetV2Backbone_human,
    'mobilenetv3': MobileNetV3Backbone,
    'resnet50': ResnetBackbone,
    'resnet34': ResnetBackbone_r34,
    'swin_transformer': SwinT,
}
