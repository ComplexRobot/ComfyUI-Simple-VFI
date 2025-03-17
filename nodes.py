import torch
import torch.nn.functional as F

from nodes import ImageScale
from comfy_extras.nodes_post_processing import Blur, Sharpen
import comfy.model_management as mm


class Simple_Frame_Interpolation:
    scale_methods = ["nearest", "nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE", ),
            "scale_method": (s.scale_methods, {"default": "nearest-exact"}),
            "multiplier": ("FLOAT", {"default": 0.5, "min": 0.01, "step": 0.01}),
            "batch_size": ("INT", { "default": 15, "min": 1, "step": 1 }),
            "gaussian_blur": ("BOOLEAN", {"default": True}),
            "blur_radius": ("INT", { "default": 2, "min": 1, "step": 1 }),
            "blur_sigma": ("FLOAT", { "default": 0.3, "min": -10.0, "max": 10.0, "step": 0.1}),
            "sharpen_alpha": ("FLOAT", { "default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01 }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"

    CATEGORY = "Simple-VFI"

    @torch.no_grad()
    def process(self, images, scale_method, multiplier, batch_size, gaussian_blur, blur_radius, blur_sigma, sharpen_alpha):
        device = mm.get_torch_device()
        B, H, W, C = images.shape
        new_frame_count = max(round(B * multiplier), 1)

        images = images.flatten(0, 2).unflatten(0, (1, B, H * W))
        images, = ImageScale.upscale(self, images, scale_method, H * W, new_frame_count, crop="disabled")

        if gaussian_blur and blur_radius != 0 and blur_sigma != 0:
            B2, H2, W2, C2 = images.shape

            images = images.permute(0, 2, 1, 3)
            images = images.flatten(0, 2).unflatten(0, (W2, H2, 1))

            for i in range(0, W2, batch_size):
                end = min(i + batch_size, W2)
                current_image = images[i:end, :, :, :]
                current_image = current_image.permute(0, 3, 1, 2).contiguous().to(device)
                current_image = F.interpolate(current_image, size=(H2, 2 * blur_radius + 1), mode='nearest-exact')
                current_image = current_image.permute(0, 2, 3, 1)
                print(f"{i} -> {current_image.shape}")

                if blur_sigma > 0:
                    current_image, = Blur.blur(self, current_image, blur_radius, blur_sigma)
                elif blur_sigma < 0 :
                    current_image, = Sharpen.sharpen(self, current_image, blur_radius, -blur_sigma, sharpen_alpha)

                images[i:end] = current_image[:, :, blur_radius:-blur_radius, :].cpu()

            images = images.flatten(0, 2).unflatten(0, (1, W2, H2))
            images = images.permute(0, 2, 1, 3)

        images = images.flatten(0, 2).unflatten(0, (new_frame_count, H, W))

        return (images,)


NODE_CLASS_MAPPINGS = {
    "Simple_Frame_Interpolation": Simple_Frame_Interpolation,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Simple_Frame_Interpolation": "Simple Frame Interpolation (VFI)",
}