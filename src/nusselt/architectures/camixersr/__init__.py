
import math
import re

from .camixersr import camixersr
from ...registry.model_descriptor import StateDict, ModelDescriptor

def load(state_dict: StateDict) -> ModelDescriptor[camixersr]:
    state_dict_keys = state_dict.keys()
    layer_dict = {}
    for r in state_dict_keys:
        g = re.match(r"body.(\d*).body.(\d*).norm1", r)
        if g:
            num_layer, num_block = g.groups()
            layer_dict[int(num_layer) + 1] = int(num_block) + 1

    n_group = len(layer_dict)
    n_block = [i for i in layer_dict.values()]
    ratio = 0.5  # I don't know if it's even possible to get it
    n_feats = state_dict["head.weight"].shape[0]
    model_tail = state_dict["tail.0.weight"].shape
    n_colors = model_tail[3]
    scale = int(math.sqrt(model_tail[0] / n_colors))

    model = camixersr(
        scale=scale,
        n_feats=n_feats,
        n_colors=n_colors,
        n_block=n_block,
        n_group=n_group,
        ratio=ratio

    )

    return ModelDescriptor(
        model,
        state_dict,
        architecture="camixersr",
        scale=scale,
        input_channels=n_colors,
        output_channels=n_colors
    )
