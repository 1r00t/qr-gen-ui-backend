import base64
import requests

from fastapi import FastAPI, HTTPException
from PIL import Image, PngImagePlugin
from time import time_ns
from io import BytesIO
from pydantic import BaseModel

from payload.base import PayloadTxt2Img
from payload.controlnet import ControlNetArgs
from qrgen.generator import QRCodeGenerator


app = FastAPI()


class GenerateImageRequest(BaseModel):
    qr_code_input: str
    transform_amount: int = 1
    qr_version: int = 4
    steps: int = 22
    enable_hr: bool = False
    low_vram: bool = False
    hr_scale: float = 0
    prompt: str
    negative_prompt: str


@app.post("/generate_image")
def generate_image(request: GenerateImageRequest):
    qr_code_input = request.qr_code_input
    transform_amount = request.transform_amount
    qr_version = request.qr_version
    steps = request.steps
    enable_hr = request.enable_hr
    low_vram = request.low_vram
    hr_scale = request.hr_scale
    prompt = request.prompt
    negative_prompt = request.negative_prompt

    # Generate QR Code
    generator = QRCodeGenerator(
        input_string=qr_code_input,
        version=qr_version,
        transform_amount=transform_amount,
    )
    qr_code_image = generator.generate_qr_code()
    buffered = BytesIO()
    qr_code_image.save(buffered, format="PNG")
    qr_code_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare payload
    payload = PayloadTxt2Img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        enable_hr=enable_hr,
        low_vram=low_vram,
        hr_scale=hr_scale,
    )

    # Prepare controlnets
    controlnets = [
        ControlNetArgs(
            model="control_v11f1e_sd15_tile [a371b31b]",
            module="tile_resample",
            weight=0.87,
            pixel_perfect=True,
            guidance_start=0.23,
            guidance_end=0.9,
            input_image=qr_code_image_base64,
        ),
        # ControlNetArgs(
        # model="control_v1p_sd15_brightness [5f6aa6ed]",
        #     module=None,
        #     weight=0.7,
        #     pixel_perfect=True,
        #     guidance_start=0.5,
        #     guidance_end=0.9,
        #     input_image=qr_code_image,
        # ),
    ]

    # Add controlnets to payload
    for controlnet in controlnets:
        payload.add_controlnet(controlnet)

    # Send payload to API
    API_URL = "http://localhost:7860"
    response = requests.post(f"{API_URL}/sdapi/v1/txt2img", json=payload.get())

    # Save response images
    if response.status_code == 200:
        images = response.json()["images"]
        output_image = Image.open(BytesIO(base64.b64decode(images[0])))

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", response.json()["info"])

        filename = "".join(
            [c for c in prompt if c.isalpha() or c.isdigit() or c == " "]
        ).strip()

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        output_image.save(
            f"./output/{time_ns() // 100000000} {filename}.png", pnginfo=metadata
        )
        output_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": output_image_base64}

    else:
        if response.json()["errors"]:
            raise HTTPException(status_code=500, detail=response.json()["errors"])
        else:
            raise HTTPException(
                status_code=500, detail="An error occurred during image generation."
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
