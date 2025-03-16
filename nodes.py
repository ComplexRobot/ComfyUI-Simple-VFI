import torch

from nodes import ImageScale

class Simple_Frame_Interpolation:
    scale_methods = ["nearest", "nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE", ),
            "scale_method": (s.scale_methods, {"default": "bislerp"}),
            "multiplier": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 20.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"

    CATEGORY = "Simple-VFI"

    @torch.no_grad()
    def process(self, images, scale_method, multiplier):
        B, H, W, C = images.shape
        new_frame_count = max(round(B * multiplier), 1)

        images = images.flatten(0, 2).unflatten(0, (1, B, H * W))
        images, = ImageScale.upscale(self, images, scale_method, H * W, new_frame_count, crop="disabled")
        images = images.flatten(0, 2).unflatten(0, (new_frame_count, H, W))

        return (images,)


NODE_CLASS_MAPPINGS = {
    "Simple_Frame_Interpolation": Simple_Frame_Interpolation,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Simple_Frame_Interpolation": "Simple Frame Interpolation (VFI)",
}