from nusselt.architectures.omnisr import load, OmniSR
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_omnisr_load():
    assert_loads_correctly(
        load,
        lambda: OmniSR(),
        lambda: OmniSR(num_in_ch=1, num_out_ch=3),
        lambda: OmniSR(num_in_ch=1, num_out_ch=1),
        lambda: OmniSR(num_in_ch=3, num_out_ch=3),
        lambda: OmniSR(num_in_ch=4, num_out_ch=4),
        lambda: OmniSR(bias=False),
        lambda: OmniSR(num_feat=32),
        lambda: OmniSR(block_num=2),
        lambda: OmniSR(pe=False),
        lambda: OmniSR(window_size=16),
        lambda: OmniSR(up_scale=1),
        condition=lambda a, b: (a.up_scale == b.up_scale and a.res_num == b.res_num and a.window_size == b.window_size),
    )


def test_omnisr_inference(snapshot):
    file = ModelFile(name="net_g_20000.pth")
    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)

    assert_image_inference(
        file,
        model,
        [TestImage.GRAY_128],
    )
