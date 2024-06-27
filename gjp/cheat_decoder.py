from typing import Sequence

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn

from .model import MLP


def round_smoothing(x):
    return x - jnp.sin(x * 2 * jnp.pi) / (4 * jnp.pi)


class CheatDecoder(nn.Module):
    arch_mlp_stack
    pass
