from typing import Dict, List
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# class Payload:

#     def __init__(
#         self,
#         prompt: str,
#         negative_prompt: str,
#         steps: int = 22,
#         width: int = 512,
#         height: int = 512,
#         low_vram: bool = False,
#         batch_size: int = 1,
#         n_iter: int = 1,
#     ) -> None:
#         self.payload = {}
#         self.payload["prompt"] = prompt
#         self.payload["negative_prompt"] = negative_prompt
#         self.payload["steps"] = steps
#         self.payload["width"] = width
#         self.payload["height"] = height
#         self.payload["batch_size"] = batch_size
#         self.payload["n_iter"] = n_iter
#         self.low_vram = low_vram

#     def add_controlnet(self, controlnet):
#         controlnet.low_vram = self.low_vram
#         self.payload["alwayson_scripts"]["controlnet"]["args"].append(
#             controlnet.payload
#         )

#     def get(self):
#         return self.payload


# class PayloadTxt2Img(Payload):
#     def __init__(
#         self,
#         prompt: str,
#         negative_prompt: str,
#         steps: int = 22,
#         width: int = 512,
#         height: int = 512,
#         enable_hr: bool = False,
#         hr_scale: float = 0,
#         low_vram: bool = False,
#     ) -> None:
#         with open("payload/payload_txt2img.json", "r") as payload_template_fp:
#             self.payload = json.load(payload_template_fp)
#         super().__init__(prompt, negative_prompt, steps, width, height, low_vram)
#         self.payload["enable_hr"] = enable_hr
#         self.payload["hr_scale"] = hr_scale


# class PayloadImg2Img(Payload):
#     def __init__(
#         self,
#         prompt: str,
#         negative_prompt: str,
#         init_images: List[str],
#         steps: int = 22,
#         width: int = 512,
#         height: int = 512,
#         low_vram: bool = False,
#         denoising_strength: float = 7.5,
#     ) -> None:
#         with open("payload/payload_img2img.json", "r") as payload_template_fp:
#             self.payload = json.load(payload_template_fp)
#         super().__init__(prompt, negative_prompt, steps, width, height, low_vram)

#         self.payload["init_images"] = init_images
#         self.payload["denoising_strength"] = denoising_strength


class Payload:
    prompt: str = ""
    negative_prompt: str = ""

    def __init__(self, template: Dict, overwrites: Dict | None = None) -> None:
        self.data = template

        for key, value in overwrites.items():
            if key in self.data.keys():
                if key == "prompt":
                    self.prompt = value
                if key == "negative_prompt":
                    self.negative_prompt = value
                self.data[key] = value

    def get(self):
        self.data["prompt"] = self.prompt
        self.data["negative_prompt"] = self.negative_prompt

        return self.data

    def add_controlnet(self, controlnet: Dict):
        self.data["alwayson_scripts"]["controlnet"]["args"].append(controlnet.payload)
