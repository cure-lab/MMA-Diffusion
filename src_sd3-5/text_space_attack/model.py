from diffusers import StableDiffusion3Pipeline
import torch


MODEL_NAME = {
    "stable-diffusion-3.5-medium": "stabilityai/stable-diffusion-3.5-medium",
}


def get_model_path(model_name):
    return MODEL_NAME.get(model_name, model_name)


class GenerativeModel:
    """SD3.5 generative model wrapper."""

    def __init__(
        self,
        model_path: str = "stable-diffusion-3.5-medium",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        use_safe_checker: bool = False,
    ):
        self.model_path = get_model_path(model_path)
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_safe_checker = use_safe_checker

        self.model = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
        ).to(device)

    def generate(self, prompt: str, **kwargs):
        return self.model(
            prompt,
            num_inference_steps=kwargs.get("num_inference_steps", 28),
            guidance_scale=kwargs.get("guidance_scale", 7.0),
        )
