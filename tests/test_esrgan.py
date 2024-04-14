from nusselt.architectures.esrgan import load, RRDBNet
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_esrgan_load():
    assert_loads_correctly(
        load,
        lambda: RRDBNet(),
        lambda: RRDBNet(in_nc = 1,out_nc=3),
        lambda: RRDBNet(in_nc = 1, out_nc=1),
        lambda: RRDBNet(in_nc = 3, out_nc=3),
        lambda: RRDBNet(in_nc = 4, out_nc=4),
        lambda: RRDBNet(num_filters = 64),
        lambda: RRDBNet(num_blocks = 23),
        lambda: RRDBNet(scale = 4),
        lambda: RRDBNet(plus = False),
        lambda: RRDBNet(shuffle_factor = 4),
        condition=lambda a, b: (a.scale == b.scale),

    )


def test_esrgan_inference(snapshot):
    file = ModelFile(name="4x-UltraMix_Balanced.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
