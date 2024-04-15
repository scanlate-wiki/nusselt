from nusselt.architectures.camixersr import load, camixersr
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_camixersr_load():
    assert_loads_correctly(
        load,
        lambda: camixersr(),
        lambda: camixersr(n_block = [3,3,3,1,3],n_group=5),
        lambda: camixersr(scale=1),
        lambda: camixersr(scale=2),
        lambda: camixersr(scale=3),
        lambda: camixersr(scale=4),
        lambda: camixersr(n_feats=32),
        condition=lambda a, b: (a.scale == b.scale),

    )


def test_camixersr_inference(snapshot):
    file = ModelFile(name="CAMixerSRx4.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, camixersr)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
