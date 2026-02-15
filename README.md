# ComfyUI-MotionCapture

<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-MotionCapture/">
<img src="https://pozzettiandrea.github.io/ComfyUI-MotionCapture/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-MotionCapture/">View Live Test Gallery →</a></b>
</div>

A ComfyUI custom node package for GVHMR-based 3D human motion capture from video. Extract SMPL parameters and 3D skeletal motion using state-of-the-art pose estimation.



https://github.com/user-attachments/assets/17638ca5-8139-40ca-b215-0d7dabf0ea73


https://github.com/user-attachments/assets/ba7a7797-713d-4750-9210-ff07bcc6bb01

## Installation

Please always install through ComfyUI-Manager

## Credits

### GVHMR
```
@inproceedings{qiu2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Qiu, Zehong and Wang, Qingshan and Peng, Zhenbo and others},
  booktitle={SIGGRAPH Asia},
  year={2024}
}
```

### Components
- **ViTPose**: https://github.com/ViTAE-Transformer/ViTPose
- **HMR2**: https://github.com/shubham-goel/4D-Humans
- **SMPL**: https://smpl.is.tue.mpg.de/
- **SMPL-X**: https://smpl-x.is.tue.mpg.de/

## License

This package vendors GVHMR code which has its own license. Please check the original GVHMR repository for licensing terms.

SMPL and SMPL-X body models require separate licenses from their respective providers.