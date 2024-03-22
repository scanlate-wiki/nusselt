from __future__ import annotations
import os
import re
import gdown
import numpy as np
import torch
import sys

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from inspect import getsource
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import unquote, urlparse
from syrupy.filters import props

from nusselt import ModelLoader, StateDict, ModelDescriptor, ImageTransformer

MODEL_DIR = Path("./tests/models/")
IMAGE_DIR = Path("./tests/images/")


def get_url_file_name(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def download_file(url: str, filename: Path | str) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading %s to %s", url, filename)

    if "drive.google.com" in url:
        gdown.download(url=url, output=filename.absolute().as_posix(), quiet=False, fuzzy=True)
    else:
        torch.hub.download_url_to_file(url, filename.absolute().as_posix())


def get_test_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelFile:
    name: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.name

    def exists(self) -> bool:
        return self.path.exists()

    def load_model(self) -> ModelDescriptor:
        return ModelLoader().load_from_file(self.path)

    @staticmethod
    def from_url(url: str, name: str | None = None):
        file = ModelFile(name or get_url_file_name(url))

        if not file.exists():
            download_file(url, file.path)

        return file


@contextmanager
def expect_error(snapshot):
    try:
        yield None
        did_error = False
    except Exception as e:
        did_error = True
        assert e == snapshot

    if not did_error:
        raise AssertionError("Expected an error, but none was raised")


class TestImage(Enum):
    COLOR_64 = "64x64"
    GRAY_128 = "128x128g"

    __test__ = False


def assert_image_inference(
    model_file: ModelFile,
    model: ModelDescriptor,
    test_images: list[TestImage],
):
    test_images.sort(key=lambda image: image.value)

    update_mode = "--snapshot-update" in sys.argv

    model.to(get_test_device())

    for test_image in test_images:
        path = os.path.join(IMAGE_DIR, "input", test_image.value + ".png")

        image = ImageTransformer.read_image(path, "grayscale" if model.input_channels == 1 else "color")
        tensor = ImageTransformer.img2tensor(image).to(get_test_device()).unsqueeze(0)

        _, image_c, image_h, image_w = tensor.shape

        assert (
            image_c == model.input_channels
        ), f"Expected the input image '{test_image.value}' to have {model.input_channels} channels, but it had {image_c} channels."

        try:
            model.eval()
            with torch.no_grad():
                output = model(tensor)

        except Exception as e:
            raise AssertionError(f"Failed on {test_image.value}: {e}") from e
        _, output_c, output_h, output_w = output.shape

        assert (
            output_c == model.output_channels
        ), f"Expected the output of '{test_image.value}' to have {model.output_channels} channels, but it had {output_c} channels."
        assert (
            output_w == image_w * model.scale and output_h == image_h * model.scale
        ), f"Expected the input image '{test_image.value}' {image_w}x{image_h} to be scaled {model.scale}x, but the output was {output_w}x{output_h}."

        output_image = ImageTransformer.tensor2img(output)
        expected_path = IMAGE_DIR / "output" / test_image.value / f"{model_file.path.stem}.png"

        if update_mode and not expected_path.exists():
            ImageTransformer.write_image(output_image, expected_path.absolute().as_posix())
            continue

        assert expected_path.exists(), f"Expected {expected_path} to exist."
        expected = ImageTransformer.read_image(expected_path.absolute().as_posix(), float32=False)

        if model.input_channels == 1:
            close_enough = np.allclose(output_image, expected[:, :, 0], atol=1)
        else:
            close_enough = np.allclose(output_image, expected, atol=1)

        if update_mode and not close_enough:
            ImageTransformer.write_image(output_image, expected_path.absolute().as_posix())
            continue

        assert close_enough, f"Failed on {test_image.value}"


T = TypeVar("T", bound=torch.nn.Module)


def _get_different_keys(a: Any, b: Any, keys: list[str]) -> str:
    lines: list[str] = []

    keys = list(set(dir(a)).intersection(keys))
    keys.sort()

    for key in keys:
        a_val = getattr(a, key)
        b_val = getattr(b, key)
        if a_val == b_val:
            lines.append(f"{key}: {a_val}")
        else:
            lines.append(f"{key}: {a_val} != {b_val}")

    return "\n".join(lines)


def _get_compare_keys(condition: Callable) -> list[str]:
    pattern = re.compile(r"a\.(\w+)")
    return [m.group(1) for m in pattern.finditer(getsource(condition))]


def assert_loads_correctly(
    load: Callable[[StateDict], ModelDescriptor[T]],
    *models: Callable[[], T],
    condition: Callable[[T, T], bool] = lambda _a, _b: True,
):
    for model_fn in models:
        model_name = getsource(model_fn)
        try:
            model = model_fn()
        except Exception as e:
            raise AssertionError(f"Failed to create model: {model_name}") from e

        try:
            state_dict = model.state_dict()
            loaded = load(state_dict)
        except Exception as e:
            raise AssertionError(f"Failed to load: {model_name}") from e

        assert type(loaded.model) == type(
            model
        ), f"Expected {model_name} to be loaded correctly, but found a {type(loaded.model)} instead."

        assert condition(model, loaded.model), (
            f"Failed condition for {model_name}."
            f" Keys:\n\n{_get_different_keys(model, loaded.model, _get_compare_keys(condition))}"
        )


disallowed_props = props("model", "state_dict", "device", "dtype")
