import numpy as np
import torch 
from PIL import Image
from torchvision import transforms
from model.semseg.dpt import DPT
from dataset.transform import resize_fix, normalize

def filter_sd(sd):
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v 

    return new_sd

def define_dinov2():
    load_state_dict = "weight/latest.pth"

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs['small'], 'nclass': 2})

    sd = torch.load(load_state_dict)

    sd = sd["model"]
    new_sd = filter_sd(sd)
    model.load_state_dict(new_sd)
    return model 

def inference_one_image_unlabeled(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h = image_np.shape[0]
    w = image_np.shape[1]
    size = (h, w)

    image = resize_fix(image, size=518)
    image = normalize(image)[None, ]
    image = image.to(device)
    with torch.no_grad():
        output = model(image)

    output = output.argmax(dim=1)      

    to_pil = transforms.ToPILImage()

    output = output[0, ].float()
    output = np.array(
                transforms.Resize(size)(to_pil(output)))

    Image.fromarray(output).save("./example.png")


if __name__ == "__main__":
    device = f"cuda:0"
    model = define_dinov2().eval().to(device)

    inference_one_image_unlabeled("./example.jpg", 
                                  model, device)

