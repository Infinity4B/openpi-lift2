import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_lift2_example(prompt: str = "perform task") -> dict:
    """Creates a random input example for the LIFT2 policy."""
    return {
        "observation.images.head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.random.rand(14),  # 14-dim EEF state
        "prompt": prompt,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Lift2Inputs(transforms.DataTransformFn):
    """Input transform for LIFT2 dual-arm robot with 3 cameras and 14-dim state."""
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # State is the 14-dim EEF state
        state = np.asarray(data["observation.state"])

        # Parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        head_image = _parse_image(data["observation.images.head"])
        left_wrist_image = _parse_image(data["observation.images.left_wrist"])
        right_wrist_image = _parse_image(data["observation.images.right_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # Use head as base, and both wrist cameras
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (head_image, left_wrist_image, right_wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                # For FAST model, use head as base_0, left_wrist as base_1, right_wrist as wrist_0
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (head_image, left_wrist_image, right_wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "action" in data:
            inputs["actions"] = np.asarray(data["action"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Lift2Outputs(transforms.DataTransformFn):
    """Output transform for LIFT2 dual-arm robot (14-dim action space)."""
    def __call__(self, data: dict) -> dict:
        # Return all 14 dims for bimanual robot
        return {"actions": np.asarray(data["actions"][:, :14])}
