from nusselt.architectures.swinir import load, SwinIR
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_swinir_load():
    assert_loads_correctly(
        load,
        lambda: SwinIR(window_size=8),
        lambda: SwinIR(depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=8),
        lambda: SwinIR(depths=[6, 6, 2, 1], num_heads=[6, 4, 6, 3], window_size=8),
    )


def test_swinir_inference(snapshot):
    file = ModelFile(name="2xHFA2kSwinIR-S.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model,SwinIR)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
