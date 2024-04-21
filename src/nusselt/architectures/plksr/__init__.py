import math

from .plksr import PLKSR
from ...registry.model_descriptor import StateDict, ModelDescriptor
from ...utilities.state import get_seq_len


def load(state_dict: StateDict) -> ModelDescriptor[PLKSR]:
    state_keys = state_dict.keys()

    dim = 64
    n_blocks = 28
    # CCM options
    ccm_type = 'DCCM'
    # LK Options
    kernel_size = 17
    split_ratio = 0.5
    lk_type = 'PLK'
    # LK Rep options
    use_max_kernel = True
    sparse_kernels = [5, 5, 5, 5]
    sparse_dilations = [1, 2, 3, 4]
    with_idt = False  # not detect
    # EA ablation
    use_ea = False
    # Mobile Convert options
    is_coreml = False  # not detect
    scale = 4

    num_layers = get_seq_len(state_dict, "feats")
    scale = int((state_dict[f"feats.{num_layers - 1}.weight"].shape[0] / 3) ** 0.5)
    n_blocks = num_layers - 2
    ccm_shape = state_dict["feats.1.channe_mixer.2.weight"].shape[3]
    if ccm_shape == 3:
        ccm_type = "DCCM"
    elif ccm_shape == 1:
        ccm_type = "CCM"
    dim = state_dict["feats.0.weight"].shape[0]

    if "feats.1.lk.mn_conv.weight" in state_keys:
        lk_type = "RectSparsePLK"
        split_ratio = state_dict["feats.1.lk.mn_conv.weight"].shape[0] / dim
        kernel_size = state_dict["feats.1.lk.mn_conv.weight"].shape[3]
    elif "feats.1.lk.convs.0.weight" in state_keys:
        lk_type = "SparsePLK"
        split_ratio = state_dict["feats.1.lk.convs.0.weight"].shape[0] / dim
        len_sparse_kernels = get_seq_len(state_dict, f"feats.1.lk.convs")
        kernel_size = state_dict[f"feats.1.lk.convs.{len_sparse_kernels - 1}.weight"].shape[3]
        if kernel_size != 5:
            use_max_kernel = True
            len_sparse_kernels -= 1
        sparse_kernels = [5 for _ in range(len_sparse_kernels)]
        sparse_dilations = [i + 1 for i in range(len_sparse_kernels)]
    else:
        split_ratio = state_dict["feats.1.lk.conv.weight"].shape[0] / dim
        kernel_size = state_dict["feats.1.lk.conv.weight"].shape[3]

    if "feats.1.attn.f.0.weight" in state_keys:
        use_ea = True
    model = PLKSR(
        dim=dim,
        n_blocks=n_blocks,
        scale=scale,
        ccm_type=ccm_type,
        kernel_size=kernel_size,
        split_ratio=split_ratio,
        lk_type=lk_type,
        use_max_kernel=use_max_kernel,
        sparse_kernels=sparse_kernels,
        sparse_dilations=sparse_dilations,
        with_idt=with_idt,
        use_ea=use_ea,
        is_coreml=is_coreml

    )
    return ModelDescriptor(
        model,
        state_dict,
        architecture="PLKSR",
        scale=scale,
        input_channels=3,
        output_channels=3
    )
