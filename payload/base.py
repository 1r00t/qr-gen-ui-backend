from typing import Dict, List
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Payload:
    payload: Dict = {}

    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int = 22,
        low_vram: bool = False,
    ) -> None:
        self.payload["prompt"] = prompt
        self.payload["negative_prompt"] = negative_prompt
        self.payload["steps"] = steps
        self.low_vram = low_vram

    def add_controlnet(self, controlnet):
        controlnet.low_vram = self.low_vram
        self.payload["alwayson_scripts"]["controlnet"]["args"].append(
            controlnet.payload
        )

    def get(self):
        return self.payload


class PayloadTxt2Img(Payload):
    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int = 22,
        enable_hr: bool = False,
        hr_scale: float = 0,
        low_vram: bool = False,
    ) -> None:
        with open("payload/payload_txt2img.json", "r") as payload_template_fp:
            self.payload = json.load(payload_template_fp)
        super().__init__(prompt, negative_prompt, steps, low_vram)
        self.payload["enable_hr"] = enable_hr
        self.payload["hr_scale"] = hr_scale


class PayloadImg2Img(Payload):
    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        init_images: List[str],
        steps: int = 22,
        low_vram: bool = False,
        denoising_strength: float = 7.5,
    ) -> None:
        with open("payload/payload_img2img.json", "r") as payload_template_fp:
            self.payload = json.load(payload_template_fp)
        super().__init__(prompt, negative_prompt, steps, low_vram)

        self.payload["init_images"] = init_images
        self.payload["denoising_strength"] = denoising_strength
