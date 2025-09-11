from diffusers import StableDiffusionPipeline
import torch
MODEL_NAME = {
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
}


def get_model_path(model_name):
    return MODEL_NAME[model_name]
class GenerativeModel():
    def __init__(
        self,
        model_path: str="runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        use_safe_checker:bool =False
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        if use_safe_checker:
            self.model = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.torch_dtype)
        else:
            self.model = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.torch_dtype,safety_checker=None)
        self.model=self.model.to(device)
        
    def generate(self, prompt:str):
        results = self.model(prompt)
        return results  