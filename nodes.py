import torch
import torch.nn.functional as functional
from tqdm import tqdm

from nodes import ImageScale
import comfy.model_management as model_management
from comfy_extras.nodes_post_processing import Blur, Sharpen
from comfy.utils import ProgressBar


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
        device = model_management.get_torch_device()
        blur_enabled = gaussian_blur and blur_radius != 0 and blur_sigma != 0
        scale_batch_size = batch_size * (2 * blur_radius + 1 if blur_enabled else 1)
        B, H, W, C = images.shape
        new_frame_count = max(round(B * multiplier), 1)

        # convert batch into one large image with horizontal dimension = flattened image, vertical dimension = time
        images = images.reshape(1, B, H * W, C)
        # transpose for correct memory layout
        images = images.permute(0, 2, 1, 3)
        # reshape into batches where batch = single image pixel, vertical = time, horizontal = nothing (interpolate time only)
        images = images.reshape(H * W, B, 1, C)

        # scale images on the time axis (increase or reduce frame count)
        progress_bar = ProgressBar(H * W * (2 if blur_enabled else 1))
        scaled_list = []
        for current_image in tqdm(images.split(scale_batch_size), f"{scale_method.capitalize()} VFI"):
            current_image = current_image.contiguous().to(device)
            # for lanczos and bislerp, it is faster to pass as a single unbatched image
            if scale_method == "lanczos" or scale_method == "bislerp":
                current_image = current_image.reshape(1, current_image.size(0), B, C)
                current_image = current_image.permute(0, 2, 1, 3)
                current_image, = ImageScale.upscale(self, current_image, scale_method, current_image.size(-2), new_frame_count, crop="disabled")
                current_image = current_image.permute(0, 2, 1, 3)
                current_image = current_image.reshape(current_image.size(1), new_frame_count, 1, C)
            else:
                current_image, = ImageScale.upscale(self, current_image, scale_method, 1, new_frame_count, crop="disabled")
            scaled_list += current_image.cpu().split(1)
            progress_bar.update(current_image.size(0))
            model_management.throw_exception_if_processing_interrupted()
        images = torch.stack(scaled_list).squeeze(1).cpu()

        if blur_enabled:
            B2, H2, W2, C2 = images.shape
            
            blurred_list = []
            for current_image in tqdm(images.split(batch_size), f"{"Gaussian blur" if blur_sigma > 0 else "Sharpen"} VFI"):
                # add padding so the reflect padding added by blur/sharpen will succeed (inefficient!)
                current_image = current_image.permute(0, 3, 1, 2).contiguous().to(device)
                current_image = functional.interpolate(current_image, size=(H2, 2 * blur_radius + 1), mode='nearest-exact')
                current_image = current_image.permute(0, 2, 3, 1)

                if blur_sigma > 0:
                    current_image, = Blur.blur(self, current_image, blur_radius, blur_sigma)
                elif blur_sigma < 0:
                    current_image, = Sharpen.sharpen(self, current_image, blur_radius, -blur_sigma, sharpen_alpha)

                blurred_list += current_image[:, :, blur_radius:-blur_radius, :].cpu().split(1)
                progress_bar.update(current_image.size(0))
                model_management.throw_exception_if_processing_interrupted()
            images = torch.stack(blurred_list).squeeze(1).cpu()

        # invert initial operations
        images = images.reshape(1, H * W, new_frame_count, C)
        images = images.permute(0, 2, 1, 3)
        images = images.reshape(new_frame_count, H, W, C)

        return (images,)


NODE_CLASS_MAPPINGS = {
    "Simple_Frame_Interpolation": Simple_Frame_Interpolation,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Simple_Frame_Interpolation": "Simple Frame Interpolation (VFI)",
}