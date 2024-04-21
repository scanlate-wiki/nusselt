from nusselt.architectures.dat import (load, DAT)
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
        lambda: DAT(),
        lambda: DAT(embed_dim=60),
        lambda: DAT(in_chans=1),
        lambda: DAT(in_chans=4),
        lambda: DAT(depth=[2, 3], num_heads=[2, 5]),
        lambda: DAT(depth=[2, 3, 4, 2], num_heads=[2, 3, 2, 2]),
        lambda: DAT(depth=[2, 3, 4, 2, 5], num_heads=[2, 3, 2, 2, 3]),
        lambda: DAT(upsampler="pixelshuffle", upscale=1),
        lambda: DAT(upsampler="pixelshuffle", upscale=2),
        lambda: DAT(upsampler="pixelshuffle", upscale=3),
        lambda: DAT(upsampler="pixelshuffle", upscale=4),
        lambda: DAT(upsampler="pixelshuffle", upscale=8),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=1),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=2),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=3),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=4),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=8),
        lambda: DAT(resi_connection="3conv"),
        lambda: DAT(qkv_bias=False),
        lambda: DAT(split_size=[4, 4]),
        lambda: DAT(split_size=[2, 8]),
    )


def test_dat_inference(snapshot):
    file = ModelFile(name="4x_IllustrationJaNai_V1_DAT2_190k.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model,DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
