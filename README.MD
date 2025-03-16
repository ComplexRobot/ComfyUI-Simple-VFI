# ComfyUI Simple VFI

Nodes for basic frame interpolation without the use of AI.

## Table of Contents

* [Nodes](#nodes)
  * [Simple Frame Interpolation (VFI)](#simple-frame-interpolation-vfi)

## Nodes

### Simple Frame Interpolation (VFI)

A node for doing very simple frame interpolation operations.
Uses basic operations (bicubic, lanczos, etc.) to blend the frames together.

|              |                                                                          |
|--------------|--------------------------------------------------------------------------|
| images       | Image frames to interpolate.                                             |
| scale_method | Method used to interpolate the frames.                                   |
| multiplier   | Value to multiply the number of frames by. Not limited to whole numbers. |