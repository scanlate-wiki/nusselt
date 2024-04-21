# nusselt

## Usage
```python
from nusselt import ModelLoader, ImageTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModelLoader(device).load_from_file("4x_sr_model.pth")
model.eval()

image = ImageTransformer.read_image("input.png", "grayscale" if model.input_channels == 1 else "color")
output = ImageTransformer.img2tensor(image).to(device)

with torch.no_grad():
    output = model(output)

output_image = ImageTransformer.tensor2img(output)
ImageTransformer.write_image(output_image, "output.png")
```

## Architectures
* [SPAN](https://github.com/hongyuanyu/span)
* [DITN](https://github.com/yongliuy/DITN)
* [OmniSR](https://github.com/Francis0625/Omni-SR)
* [DAT](https://github.com/zhengchen1999/dat)
* [SwinIr](https://github.com/JingyunLiang/SwinIR)
* [Esrgan](https://github.com/xinntao/Real-ESRGAN)
* [Real-Cugan](https://github.com/bilibili/ailab)
* [CAMixerSR](https://github.com/icandle/CAMixerSR)
* [SwinIR](https://github.com/JingyunLiang/SwinIR)
* [RGT](https://github.com/zhengchen1999/RGT)
* [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary)
## Credits
* Based on [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel)
