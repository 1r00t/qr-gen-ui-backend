import json


class ControlNetArgs:
    input_image: str = ""
    module: str | None = None
    model: str = ""
    weight: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    pixel_perfect: bool = False
    lowvram: bool = False

    def __init__(
        self,
        input_image: str,
        module: str,
        model: str,
        weight: float,
        guidance_start: float,
        guidance_end: float,
        pixel_perfect: bool,
    ) -> None:
        self.input_image = input_image
        self.module = module
        self.model = model
        self.weight = weight
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.pixel_perfect = pixel_perfect

    @property
    def payload(self) -> str:
        return {
            "input_image": self.input_image,
            "module": self.module,
            "model": self.model,
            "weight": self.weight,
            "guidance_start": self.guidance_start,
            "guidance_end": self.guidance_end,
            "pixel_perfect": self.pixel_perfect,
            "lowvram": self.lowvram,
        }
