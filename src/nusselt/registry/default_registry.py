from .architecture_registry import ArchRegistry, ArchSupport
from ..architectures import span, ditn, omnisr, esrgan

MAIN_REGISTRY = ArchRegistry()

MAIN_REGISTRY.add(
    ArchSupport(
        id="SPAN",
        keys=["block_1.c1_r.sk.weight", "block_1.c2_r.sk.weight", "conv_cat.weight"],
        load=span.load,
    ),
    ArchSupport(
        id="DITN",
        keys=[
            "sft.weight",
            "UFONE.0.ITLs.0.attn.temperature",
            "UFONE.0.ITLs.0.ffn.project_in.weight",
            "UFONE.0.ITLs.0.ffn.dwconv.weight",
            "UFONE.0.ITLs.0.ffn.project_out.weight",
            "conv_after_body.weight",
            "upsample.0.weight",
        ],
        load=ditn.load,
    ),
    ArchSupport(
        id="OmniSR",
        keys=["residual_layer.0.residual_layer.0.layer.0.fn.0.weight"],
        load=omnisr.load,
    ),
    ArchSupport(
        id="ESRGAN1",
        keys=["model.0.weight",
              "model.1.sub.0.RDB1.conv1.0.weight"],
        load=esrgan.load,
    ),
    ArchSupport(
        id="ESRGAN2",
        keys=["conv_first.weight",
              "body.0.rdb1.conv1.weight",
              "conv_body.weight",
              "conv_last.weight"],
        load=esrgan.load,
    ),
    ArchSupport(
        id="REAL-ESRGAN",
        keys=["conv_first.weight",
              "RRDB_trunk.0.RDB1.conv1.weight",
              "trunk_conv.weight",
              "conv_last.weight"],
        load=esrgan.load,
    ),
    ArchSupport(
        id="ESRGAN+",
        keys=["model.0.weight", "model.1.sub.0.RDB1.conv1x1.weight"],
        load=esrgan.load,
    ),
)
