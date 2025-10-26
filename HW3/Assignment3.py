import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        torch_img = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)
        torch_img = torch.from_numpy(torch_img).to(torch.float32)

        return torch_img

    def brighten(self, torch_img):
        bright_img = torch_img + 100.0

        return bright_img

    def saturation_arithmetic(self, img):
        torch_img = self.torch_image_conversion(img).to(torch.uint8)
        bright_img = torch_img + 100
        saturated_img = torch.clamp(bright_img, 0, 255)

        return saturated_img

    def add_noise(self, torch_img):
        noise = torch.randn_like(torch_img) * 100.0
        noisy_img = torch_img + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 255.0) / 255.0

        return noisy_img

    def normalization_image(self, img):
        img = self.torch_image_conversion(img).to(torch.float64)

        mean = img.mean(dim=(0, 1))
        std = img.std(dim=(0, 1))

        image_norm = (img - mean) / std

        return image_norm

    def Imagenet_norm(self, img):
        img = self.torch_image_conversion(img).to(torch.float64)
        
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        ImageNet_norm = (img / 255.0 - mean) / std
        ImageNet_norm = torch.clamp(ImageNet_norm, 0.0, 1.0)

        return ImageNet_norm

    def dimension_rearrange(self, img):
        torch_img = self.torch_image_conversion(img).permute(2, 0, 1)
        rearrange = torch_img.unsqueeze(0)

        return rearrange
    
    def stride(self, img):
        torch_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float32)

        scharr_x = torch.tensor([[3, 0, -3],
                                 [10, 0, -10],
                                 [3, 0, -3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        stride_image = torch.nn.functional.conv2d(torch_img, scharr_x, stride = 2, padding = 1).squeeze(0).squeeze(0)
    
        return stride_image

    def chain_rule(self, x, y, z):

        return df_dx, df_dy, df_dz, df_dq

    def relu(self, x, w):

        return dx, dw


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    img_cat = cv.imread("cat_eye.jpg", cv.IMREAD_GRAYSCALE)
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    # plt.imsave('torch.png', torch_img/255.0)
    bright_img = assign.brighten(torch_img)
    #print(bright_img)
    saturated_img = assign.saturation_arithmetic(img)
    #plt.imsave('saturated.png', saturated_img.to(torch.float32)/255.0)
    noisy_img = assign.add_noise(torch_img)
    #plt.imsave('noisy.png', noisy_img)
    image_norm = assign.normalization_image(img)
    #print(image_norm)
    ImageNet_norm = assign.Imagenet_norm(img)
    #plt.imsave('ImageNet_norm.png', ImageNet_norm) 
    rearrange = assign.dimension_rearrange(img)
    stride_image = assign.stride(img_cat)
    #plt.imsave('stride.png', stride_image)
    df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2.0, y=5.0, z=-4.0)
    dx, dw = assign.relu(x=[-1.0, 2.0], w=[2.0, -3.0, -3.0])
