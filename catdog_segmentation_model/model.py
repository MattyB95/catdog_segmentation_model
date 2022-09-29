import os

import numpy as np
import requests
import torch
from torchvision import transforms

import unet


class CatDogUNet:
    def __init__(self):
        filename = "unet_model.ckpt"
        if not os.path.exists(filename):
            url = "https://connectionsworkshop.blob.core.windows.net/pets/conv_net_model_mse10.ckpt"
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
        self.model = unet.UNet()
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def predict(self, image):
        # pre-process input image (as required by model)
        transform_input = transforms.Compose([transforms.Resize((192, 192)), ])
        image = image.values
        image = image[:, :, 0:3]  # make sure we have only 3 channels
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        image = torch.from_numpy(image).type(torch.float32)
        image = transform_input(image)

        # make prediction
        ### TODO ###
        ### Apply the model to the pre-processed input image.
        ### Return the segmentation mask.
        return "FIX ME"


CatDogUNet()
