from nusselt.architectures.span import load, SPAN
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_span_load():
    assert_loads_correctly(
        load,
        lambda: SPAN(num_in_ch=3, num_out_ch=3),
        lambda: SPAN(num_in_ch=1, num_out_ch=3),
        lambda: SPAN(num_in_ch=1, num_out_ch=1),
        lambda: SPAN(num_in_ch=4, num_out_ch=4),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, feature_channels=32),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, feature_channels=64),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=1),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=2),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=4),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=8),
        condition=lambda a, b: (
            a.in_channels == b.in_channels and a.out_channels == b.out_channels and a.img_range == b.img_range
        ),
    )


def test_span_inference(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1mqixUK-Zgf-XW0ipma64T30o4K6Mr7_0/view?usp=sharing",
        "4x_span_anime_pretrain.pth",
    )
    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SPAN)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
