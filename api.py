import base64
import requests

from fastapi import FastAPI, HTTPException
from PIL import Image, PngImagePlugin
from time import time_ns
from io import BytesIO
from pydantic import BaseModel

from payload.base import Payload
from payload.templates import img2img
from payload.controlnet import ControlNetArgs
from qrgen.generator import QRCodeGenerator


app = FastAPI()


class GenerateImageRequest(BaseModel):
    qr_code_input: str
    transform_amount: int = 1
    code_scale: float = 0.85
    qr_version: int = 4
    steps: int = 22
    width: int = 512
    height: int = 512
    enable_hr: bool = False
    low_vram: bool = False
    n_iter: int = 1
    batch_size: int = 1
    hr_scale: float = 0
    prompt: str
    negative_prompt: str
    seed: int = -1
    subseed: int = 0.0


@app.post("/generate_image")
def generate_image(request: GenerateImageRequest):
    qr_code_input = request.qr_code_input
    transform_amount = request.transform_amount
    code_scale = request.code_scale
    qr_version = request.qr_version
    steps = request.steps
    enable_hr = request.enable_hr
    low_vram = request.low_vram
    n_iter = request.n_iter
    batch_size = request.batch_size
    hr_scale = request.hr_scale
    prompt = request.prompt
    negative_prompt = request.negative_prompt
    seed = request.seed
    subseed = request.subseed

    # Generate QR Code
    tile_generator = QRCodeGenerator(
        input_string=qr_code_input,
        version=qr_version,
        transform_amount=transform_amount,
        code_scale=code_scale,
        bg_color=(255, 255, 255),
        module_color=(0, 0, 0),
        pattern_color=(0, 0, 0),
        quiet_color=(255, 255, 255),
    )
    tile_qr_code_image = tile_generator.generate_qr_code()
    buffered = BytesIO()
    tile_qr_code_image.save(buffered, format="PNG")
    tile_qr_code_image.save("tile_qr.png", format="PNG")
    tile_qr_code_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    brightness_generator = QRCodeGenerator(
        input_string=qr_code_input,
        version=qr_version,
        transform_amount=transform_amount,
        code_scale=code_scale,
    )
    brightness_qr_code_image = brightness_generator.generate_qr_code()
    buffered = BytesIO()
    brightness_qr_code_image.save(buffered, format="PNG")
    brightness_qr_code_image.save("brightness_qr.png", format="PNG")
    brightness_qr_code_image_base64 = base64.b64encode(buffered.getvalue()).decode(
        "utf-8"
    )

    img2img_overwrites = {
        "init_images": [tile_qr_code_image_base64],
        "steps": steps,
        "width": 768,
        "height": 768,
        "low_vram": low_vram,
        "n_iter": n_iter,
        "batch_size": batch_size,
        "seed": seed,
        "subseed": subseed,
    }
    payload = Payload(template=img2img, overwrites=img2img_overwrites)

    payload.prompt = prompt
    payload.negative_prompt = negative_prompt

    # Prepare payload
    # payload = PayloadImg2Img(
    #     init_images=[tile_qr_code_image_base64],
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     steps=steps,
    #     width=768,
    #     height=768,
    #     low_vram=low_vram,
    # )
    # payload = PayloadTxt2Img(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     steps=steps,
    #     enable_hr=enable_hr,
    #     low_vram=low_vram,
    #     hr_scale=hr_scale,
    # )

    # Prepare controlnets
    controlnets = [
        ControlNetArgs(
            model="control_v11f1e_sd15_tile [a371b31b]",
            module="tile_resample",
            weight=0.67,
            pixel_perfect=True,
            guidance_start=0.23,
            guidance_end=0.9,
            input_image=tile_qr_code_image_base64,
        ),
        ControlNetArgs(
            model="control_v1p_sd15_brightness [5f6aa6ed]",
            module=None,
            weight=0.3,
            pixel_perfect=True,
            guidance_start=0.5,
            guidance_end=0.9,
            input_image=brightness_qr_code_image_base64,
        ),
    ]

    # Add controlnets to payload
    for controlnet in controlnets:
        payload.add_controlnet(controlnet)

    # Send payload to API
    API_URL = "http://localhost:7860"
    response = requests.post(f"{API_URL}/sdapi/v1/img2img", json=payload.get())

    # Save response images
    if response.status_code == 200:
        images = response.json()["images"]

        for image in images[: n_iter * batch_size]:
            output_image = Image.open(BytesIO(base64.b64decode(image)))

            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("parameters", response.json()["info"])

            filename = "".join(
                [c for c in prompt if c.isalpha() or c.isdigit() or c == " "]
            ).strip()[:80]

            buffered = BytesIO()
            output_image.save(buffered, format="PNG")
            output_image.save(
                f"./output/{time_ns() // 100000000} {filename}.png", pnginfo=metadata
            )
            # output_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"success": True}

    else:
        if response.text:
            raise HTTPException(status_code=500, detail=response.text)
        if response.json()["errors"]:
            raise HTTPException(status_code=500, detail=response.json()["errors"])
        elif response.json()["error"]:
            raise HTTPException(status_code=500, detail=response.json()["error"])
        else:
            raise HTTPException(
                status_code=500, detail="An error occurred during image generation."
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
