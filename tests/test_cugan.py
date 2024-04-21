from nusselt.architectures.cugan import (load, cugan)
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_cugan_load():
    assert_loads_correctly(
        load,
        lambda: cugan(),
        lambda: cugan(in_channels=1, out_channels=3),
        lambda: cugan(in_channels=3, out_channels=1),
        lambda: cugan(in_channels=3, out_channels=3),
        lambda: cugan(in_channels=4, out_channels=4),
        lambda: cugan(scale=2),
        lambda: cugan(scale=3),
        lambda: cugan(scale=4),
        condition=lambda a, b: (
                a.scale == b.scale
        ),
    )


def test_cugan_inference(snapshot):
    file = ModelFile(name="2x_umzi_Mahou_cugan.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model,cugan)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
