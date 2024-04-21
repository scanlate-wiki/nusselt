from nusselt.architectures.atd import load, ATD
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_atd_load():
    assert_loads_correctly(
        load,
        lambda: ATD(),
        lambda: ATD(in_chans=4, embed_dim=60),
        lambda: ATD(window_size=4),
        lambda: ATD(depths=(4, 6, 8, 7, 5), num_heads=(4, 6, 8, 12, 5)),
        lambda: ATD(num_tokens=32, reducted_dim=3, convffn_kernel_size=7, mlp_ratio=3),
        lambda: ATD(qkv_bias=False),
        lambda: ATD(patch_norm=False),
        lambda: ATD(ape=True),
        lambda: ATD(resi_connection="1conv"),
        lambda: ATD(resi_connection="3conv"),
        lambda: ATD(upsampler="", upscale=1),
        lambda: ATD(upsampler="nearest+conv", upscale=4),
        lambda: ATD(upsampler="pixelshuffle", upscale=1),
        lambda: ATD(upsampler="pixelshuffle", upscale=2),
        lambda: ATD(upsampler="pixelshuffle", upscale=3),
        lambda: ATD(upsampler="pixelshuffle", upscale=4),
        lambda: ATD(upsampler="pixelshuffle", upscale=8),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=1),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=2),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=3),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=4),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=8),
        condition=lambda a, b: (a.upscale == b.upscale),

    )


def test_atd_inference(snapshot):
    file = ModelFile(name="003_ATD_SRx4_finetune.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ATD)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
