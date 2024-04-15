from .SRVGG import SRVGGNetCompact as Compact
from ...registry.model_descriptor import StateDict, ModelDescriptor
from ...utilities.state import get_seq_len, get_scale_and_output_channels


def load(state_dict: StateDict) -> ModelDescriptor[Compact]:
    state = state_dict

    highest_num = get_seq_len(state, "body") - 1

    in_nc = state["body.0.weight"].shape[1]
    num_feat = state["body.0.weight"].shape[0]
    num_conv = (highest_num - 2) // 2

    pixelshuffle_shape = state[f"body.{highest_num}.bias"].shape[0]
    scale, out_nc = get_scale_and_output_channels(pixelshuffle_shape, in_nc)

    model = Compact(
        num_in_ch=in_nc,
        num_out_ch=out_nc,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=scale,
    )

    return ModelDescriptor(
        model,
        state_dict,
        architecture="Compact",
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc
    )
