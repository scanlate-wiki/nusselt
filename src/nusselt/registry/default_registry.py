from .architecture_registry import ArchRegistry, ArchSupport
from ..architectures import span

MAIN_REGISTRY = ArchRegistry()

MAIN_REGISTRY.add(
    ArchSupport(
        id="SPAN",
        keys=["block_1.c1_r.sk.weight", "block_1.c2_r.sk.weight", "conv_cat.weight"],
        load=span.load,
    ),
)
