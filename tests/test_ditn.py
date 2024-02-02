from nusselt.architectures.ditn import load, DITN_Real as DITN
from tests.util import (
    ModelFile,
    TestImage,
    assert_loads_correctly,
    assert_image_inference,
    disallowed_props,
)


def test_ditn_load():
    assert_loads_correctly(
        load,
        lambda: DITN(),
        lambda: DITN(dim=4),
        lambda: DITN(dim=16),
        lambda: DITN(bias=True),
        lambda: DITN(upscale=8),
        lambda: DITN(UFONE_blocks=3),
        lambda: DITN(UFONE_blocks=5),
        condition=lambda a, b: (
            a.patch_size == b.patch_size
            and a.dim == b.dim
            and a.scale == b.scale
            and a.SAL_blocks == b.SAL_blocks
            and a.ITL_blocks == b.ITL_blocks
        ),
    )


def test_ditn_inference(snapshot):
    file = ModelFile.from_url(
        "https://cdn.discordapp.com/attachments/1172224141789765744/1172578855022760026/2x_AniScale2_DITN_i16_75K.pth"
    )
    model = file.load_model()

    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DITN)

    assert_image_inference(
        file,
        model,
        [TestImage.COLOR_64],
    )
