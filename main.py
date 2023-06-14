import base64
import requests

from PIL.PngImagePlugin import PngInfo
from time import time_ns
from PIL import Image
from io import BytesIO

# from payload.base import PayloadTxt2Img
from payload.controlnet import ControlNetArgs
from qrgen.generator import QRCodeGenerator


from payload.base import Payload
from payload.templates import img2img

x = Payload(template=img2img)

exit(0)

# generate QR-Code
generator = QRCodeGenerator(
    input_string="interstruct.com",
    version=4,
)
qr_code_image = generator.generate_qr_code()
buffered = BytesIO()
qr_code_image.save(buffered, format="PNG")
qr_code_image = base64.b64encode(buffered.getvalue())
qr_code_image = qr_code_image.decode("utf-8")


# prepare payload
payload = PayloadTxt2Img(
    prompt="cloth folds, intricate, high contrast, hard shadows",
    negative_prompt="EasyNegative, ugly, fake",
    steps=10,
    enable_hr=False,
    low_vram=True,
)


# prepare controlnets
controlnets = [
    ControlNetArgs(
        model="control_v11f1e_sd15_tile [a371b31b]",
        module="tile_resample",
        weight=0.87,
        pixel_perfect=True,
        guidance_start=0.23,
        guidance_end=0.9,
        input_image=qr_code_image,
    ),
    ControlNetArgs(
        model="control_v1p_sd15_brightness [5f6aa6ed]",
        module=None,
        weight=0.7,
        pixel_perfect=True,
        guidance_start=0.5,
        guidance_end=0.9,
        input_image=qr_code_image,
    ),
]

# add controlnets to payload
for controlnet in controlnets:
    payload.add_controlnet(controlnet)

# send payload to API
API_URL = "http://localhost:7860"
response = requests.post(f"{API_URL}/sdapi/v1/txt2img", json=payload.get())


# save response images
if response.status_code == 200:
    print(response.json()["info"])
    images = response.json()["images"]
    output_image = Image.open(BytesIO(base64.b64decode(images[0])))

    metadata = PngInfo()
    metadata.add_text("parameters", response.json()["info"])

    filename = "".join(
        [c for c in payload.get()["prompt"] if c.isalpha() or c.isdigit() or c == " "]
    ).strip()
    output_image.save(
        f"./output/{time_ns() // 100000000} {filename}.png", pnginfo=metadata
    )
    output_image.save("output.png", pnginfo=metadata)

else:
    if response.json()["errors"]:
        print(response.json()["errors"])
    exit(1)
