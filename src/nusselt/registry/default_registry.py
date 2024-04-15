from .architecture_registry import ArchRegistry, ArchSupport
from ..architectures import span, ditn, omnisr, esrgan, compact, cugan, dat, atd, rgt

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
    ArchSupport(
        id="Compact",
        keys=["body.0.weight", "body.1.weight", ],
        load=compact.load,
    ),
    ArchSupport(
        id="Cugan1",
        keys=['conv_final.weight'],
        load=cugan.load,
    ),
    ArchSupport(
        id="Cugan2",
        keys=['unet1.conv1.conv.0.weight'],
        load=cugan.load,
    ),

    ArchSupport(
        id="Cugan3",
        keys=["unet1.conv_bottom.weight"],
        load=cugan.load,
    ),
    ArchSupport(
        id="DAT",
        keys=["conv_first.weight",
              "before_RG.1.weight",
              "before_RG.1.bias",
              "layers.0.blocks.0.norm1.weight",
              "layers.0.blocks.0.norm2.weight",
              "layers.0.blocks.0.ffn.fc1.weight",
              "layers.0.blocks.0.ffn.sg.norm.weight",
              "layers.0.blocks.0.ffn.sg.conv.weight",
              "layers.0.blocks.0.ffn.fc2.weight",
              "layers.0.blocks.0.attn.qkv.weight",
              "layers.0.blocks.0.attn.proj.weight",
              "layers.0.blocks.0.attn.dwconv.0.weight",
              "layers.0.blocks.0.attn.dwconv.1.running_mean",
              "layers.0.blocks.0.attn.channel_interaction.1.weight",
              "layers.0.blocks.0.attn.channel_interaction.2.running_mean",
              "layers.0.blocks.0.attn.channel_interaction.4.weight",
              "layers.0.blocks.0.attn.spatial_interaction.0.weight",
              "layers.0.blocks.0.attn.spatial_interaction.1.running_mean",
              "layers.0.blocks.0.attn.spatial_interaction.3.weight",
              "layers.0.blocks.0.attn.attns.0.rpe_biases",
              "layers.0.blocks.0.attn.attns.0.relative_position_index",
              "layers.0.blocks.0.attn.attns.0.pos.pos_proj.weight",
              "layers.0.blocks.0.attn.attns.0.pos.pos1.0.weight",
              "layers.0.blocks.0.attn.attns.0.pos.pos3.0.weight",
              "norm.weight", ],
        load=dat.load,
    ),
    ArchSupport(
        id="ATD",
        keys=["relative_position_index_SA",
              "conv_first.weight",
              "conv_first.bias",
              "layers.0.residual_group.td",
              "layers.0.residual_group.layers.0.sigma",
              "layers.0.residual_group.layers.0.norm1.weight",
              "layers.0.residual_group.layers.0.norm1.bias",
              "layers.0.residual_group.layers.0.norm2.weight",
              "layers.0.residual_group.layers.0.norm2.bias",
              "layers.0.residual_group.layers.0.norm3.weight",
              "layers.0.residual_group.layers.0.norm3.bias",
              "layers.0.residual_group.layers.0.wqkv.weight",
              "layers.0.residual_group.layers.0.attn_win.relative_position_bias_table",
              "layers.0.residual_group.layers.0.attn_win.proj.weight",
              "layers.0.residual_group.layers.0.attn_win.proj.bias",
              "layers.0.residual_group.layers.0.attn_atd.scale",
              "layers.0.residual_group.layers.0.attn_atd.wq.weight",
              "layers.0.residual_group.layers.0.attn_atd.wk.weight",
              "layers.0.residual_group.layers.0.attn_atd.wv.weight",
              "layers.0.residual_group.layers.0.attn_aca.logit_scale",
              "layers.0.residual_group.layers.0.attn_aca.proj.weight",
              "layers.0.residual_group.layers.0.convffn.fc1.weight",
              "layers.0.residual_group.layers.0.convffn.fc1.bias",
              "layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.weight",
              "layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.bias",
              "layers.0.residual_group.layers.0.convffn.fc2.weight",
              "layers.0.residual_group.layers.0.convffn.fc2.bias",
              "norm.weight",
              "norm.bias", ],
        load=atd.load,
    ),
    ArchSupport(
        id="RGT",
        keys=["conv_first.weight",
              "before_RG.1.weight",
              "layers.0.blocks.0.gamma",
              "layers.0.blocks.0.norm1.weight",
              "layers.0.blocks.0.attn.qkv.weight",
              "layers.0.blocks.0.attn.proj.weight",
              "layers.0.blocks.0.attn.attns.0.rpe_biases",
              "layers.0.blocks.0.attn.attns.0.relative_position_index",
              "layers.0.blocks.0.attn.attns.0.pos.pos_proj.weight",
              "layers.0.blocks.0.mlp.fc1.weight",
              "layers.0.blocks.0.mlp.fc2.weight",
              "layers.0.blocks.0.norm2.weight",
              "norm.weight", ],
        load=rgt.load,
    ),
)
