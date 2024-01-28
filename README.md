# nusselt

## Usage
```python
from nusselt import ModelLoader, ImageTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModelLoader().load_from_file("4x_sr_model.pth")
model.to(device)
model.eval()

image = ImageTransformer.read_image("input.png", "gray" if model.input_channels == 1 else "color")
output = ImageTransformer.img2tensor(image).to(device)

with torch.no_grad():
    output = model(output)

output_image = ImageTransformer.tensor2img(output)
ImageTransformer.write_image(output_image, "output.png")
```

## Architectures
* [SPAN](https://github.com/hongyuanyu/span)
* [DITN](https://github.com/yongliuy/DITN)

## Credits
* Based on [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel)
