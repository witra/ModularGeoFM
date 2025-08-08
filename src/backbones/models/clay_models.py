from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import yaml
from box import Box
from einops import rearrange, reduce
from torch import nn, optim

from src.backbones.decoder.satmae_decoder import SatMAEHeadViT as Decoder
from src.backbones.encoder.clay_encoder import SegmentEncoder




