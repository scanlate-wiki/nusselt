from nusselt.architectures.plksr import PLKSR,load
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_plksr_load():
    assert_loads_correctly(
        load,
        lambda: PLKSR(),
        lambda:PLKSR(n_blocks=12,kernel_size=13,use_ea=False),
        lambda: PLKSR(scale=1),
        lambda: PLKSR(scale=2),
        lambda: PLKSR(scale=3),
        lambda: PLKSR(scale=4),
    )


def test_compact_inference(snapshot):
    file = ModelFile(name="net_g_80000.pth")

    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, PLKSR)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
