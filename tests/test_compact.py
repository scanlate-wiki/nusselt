from nusselt.architectures.compact import load, Compact
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_compact_load():
    assert_loads_correctly(
        load,
        lambda: Compact(),
        lambda: Compact(num_in_ch = 1, num_out_ch=3),
        lambda: Compact(num_in_ch = 1, num_out_ch=1),
        lambda: Compact(num_in_ch = 3, num_out_ch=3),
        lambda: Compact(num_in_ch = 4, num_out_ch=4),
        lambda: Compact(num_feat = 64),
        lambda: Compact(num_conv = 4),
        lambda: Compact(upscale = 4),
        lambda: Compact(act_type="prelu"),
        condition=lambda a, b: (a.upscale == b.upscale),

    )


def test_esrgan_inference(snapshot):
    file = ModelFile(name="2x_Ani4Kv2_G6i2_Compact_107500.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Compact)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
