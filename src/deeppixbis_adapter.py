import sys
sys.path.append("src")
from deeppixbis.models.liveness_net import LivenessNet
import torch.nn as nn

class LivenessNetRGB(LivenessNet):
    """Versión adaptada de LivenessNet para imágenes RGB (3 canales)."""
    def __init__(self):
        super().__init__()
        # reemplazamos la primera capa para aceptar 3 canales en lugar de 6
        first_conv = self.layer1[0]
        self.layer1[0] = nn.Conv2d(
            in_channels=3,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
