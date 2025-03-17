# ComfyUI Simple VFI

Nodes for basic frame interpolation without the use of AI.

## Table of Contents

* [Nodes](#nodes)
  * [Simple Frame Interpolation (VFI)](#simple-frame-interpolation-vfi)

## Nodes

### Simple Frame Interpolation (VFI)

A node for doing very simple frame interpolation operations.
Uses basic operations (bicubic, lanczos, etc.) to blend the frames together. 
Also can apply a gaussian blur or sharpen effect to blend frames.
Blur will apply a motion blur effect, while sharpen will reduce motion blur.

| Input          | Description                                                                    |
|----------------|--------------------------------------------------------------------------------|
| images         | Image frames to interpolate.                                                   |
| scale_method   | Method used to interpolate the frames.                                         |
| multiplier     | Value to multiply the number of frames by. Not limited to whole numbers.       |
| batch_size     | Number of pixels to operate on at a time. Adjust for a speed/memory trade-off. |
| gaussian_blur  | Enable gaussian blur/sharpen frame blending.                                   |
| blur_radius    | The spread of the blur to neighboring frames.                                  |
| blur_sigma     | Intensity of blur/sharpen. Positive values blur, negative values sharpen.      |
| sharpen_alpha  | When blur_sigma is negative, this increases the sharpen intensity.             |