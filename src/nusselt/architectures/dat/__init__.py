import math

from .dat import DAT
from ...registry.model_descriptor import StateDict, ModelDescriptor
from ...utilities.state import get_seq_len


def load(state_dict: StateDict) -> ModelDescriptor[DAT]:
    # defaults
    img_size = 64  # cannot be deduced from state dict in general
    in_chans = 3
    embed_dim = 180
    split_size = [2, 4]
    depth = [2, 2, 2, 2]
    num_heads = [2, 2, 2, 2]
    expansion_factor = 4.0
    qkv_bias = True
    upscale = 2
    img_range = 1.0
    resi_connection = "1conv"
    upsampler = "pixelshuffle"
    num_feat = 64

    in_chans = state_dict["conv_first.weight"].shape[1]
    embed_dim = state_dict["conv_first.weight"].shape[0]

    # num_layers = len(depth)
    num_layers = get_seq_len(state_dict, "layers")
    depth = [
        get_seq_len(state_dict, f"layers.{i}.blocks") for i in range(num_layers)
    ]

    # num_heads is linked to depth
    num_heads = [2] * num_layers
    for i in range(num_layers):
        if depth[i] >= 2:
            # that's the easy path, we can directly read the head count
            num_heads[i] = state_dict[
                f"layers.{i}.blocks.1.attn.temperature"
            ].shape[0]
        else:
            # because of a head_num // 2, we can only reconstruct even head counts
            key = f"layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight"
            num_heads[i] = state_dict[key].shape[0] * 2

    upsampler = (
        "pixelshuffle" if "conv_last.weight" in state_dict else "pixelshuffledirect"
    )
    resi_connection = "1conv" if "conv_after_body.weight" in state_dict else "3conv"

    if upsampler == "pixelshuffle":
        upscale = 1
        for i in range(0, get_seq_len(state_dict, "upsample"), 2):
            num_feat = state_dict[f"upsample.{i}.weight"].shape[1]
            shape = state_dict[f"upsample.{i}.weight"].shape[0]
            upscale *= int(math.sqrt(shape // num_feat))
    elif upsampler == "pixelshuffledirect":
        num_feat = state_dict["upsample.0.weight"].shape[1]
        upscale = int(
            math.sqrt(state_dict["upsample.0.weight"].shape[0] // in_chans)
        )

    qkv_bias = "layers.0.blocks.0.attn.qkv.bias" in state_dict

    expansion_factor = float(
        state_dict["layers.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim
    )

    if "layers.0.blocks.2.attn.attn_mask_0" in state_dict:
        attn_mask_0_x, attn_mask_0_y, _attn_mask_0_z = state_dict[
            "layers.0.blocks.2.attn.attn_mask_0"
        ].shape

        img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

    if "layers.0.blocks.0.attn.attns.0.rpe_biases" in state_dict:
        split_sizes = (
                state_dict["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
        )
        split_size = [int(x) for x in split_sizes]
    model = DAT(
        img_size=img_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        split_size=split_size,
        depth=depth,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        qkv_bias=qkv_bias,
        upscale=upscale,
        img_range=img_range,
        resi_connection=resi_connection,
        upsampler=upsampler,
    )

    return ModelDescriptor(
        model,
        state_dict,
        architecture="DAT",
        scale=upscale,
        input_channels=in_chans,
        output_channels=in_chans
    )
