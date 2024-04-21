from .cugan import cugan
from ...registry.model_descriptor import StateDict, ModelDescriptor

def load(state_dict: StateDict) -> ModelDescriptor[cugan]:
    if "conv_final.weight" in state_dict:
        # UpCunet4x
        scale = 4
        in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
        out_channels = 3  # hard coded in UpCunet4x
    elif state_dict["unet1.conv_bottom.weight"].shape[2] == 5:
        # UpCunet3x
        scale = 3
        in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
        out_channels = state_dict["unet2.conv_bottom.weight"].shape[0]
    else:
        # UpCunet2x
        scale = 2
        in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
        out_channels = state_dict["unet2.conv_bottom.weight"].shape[0]
    pro = False
    if list(state_dict.keys())[-1] == "unet2.conv_bottom.bias":
        pro = True
    if "pro" in state_dict:
        pro = True
        del state_dict["pro"]
    model = cugan(
        in_channels = in_channels,
        out_channels = out_channels,
        scale=scale,
        pro=pro

    )
    return ModelDescriptor(
        model,
        state_dict,
        architecture="cugan",
        scale=scale,
        input_channels=in_channels,
        output_channels=out_channels
    )
