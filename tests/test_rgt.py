from nusselt.architectures.rgt import load, RGT
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_dat_load():
    assert_loads_correctly(
        load,
        lambda: RGT(),
        lambda: RGT(in_chans=4, embed_dim=90),
        lambda: RGT(depth=[5, 6, 2, 3, 9], num_heads=[2, 6, 2, 9, 4]),
        lambda: RGT(mlp_ratio=3.0, qkv_bias=False),
        lambda: RGT(resi_connection="1conv"),
        lambda: RGT(resi_connection="3conv"),
        lambda: RGT(c_ratio=0.75),
        lambda: RGT(split_size=[16, 16]),
        lambda: RGT(split_size=[4, 4]),
        lambda: RGT(split_size=[8, 32]),
        lambda: RGT(upscale=1),
        lambda: RGT(upscale=2),
        lambda: RGT(upscale=3),
        lambda: RGT(upscale=4),
    )


def test_rgt_inference(snapshot):
    file = ModelFile(name="4xNomosUni_rgt_s_multijpg.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model,RGT)
    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
