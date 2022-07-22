import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]

# tensor_normalizer = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
# tensor_normalizer = transforms.Normalize(mean=[1], std=[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if device is not torch.device("cpu"):
#     print(device, type(device))
#     torch.cuda.set_device(device)


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):

        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def recover_image(tensor):
    image = tensor.detach().cpu().numpy()
    # image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
    # np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]


def recover_tensor(tensor):
    m = torch.tensor(cnn_normalization_mean).view(1, 3, 1, 1).to(tensor.device)
    s = torch.tensor(cnn_normalization_std).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * s + m
    return tensor.clamp(0, 1)

def preprocess_image(image, target_width=256):
    if target_width:
        t = transforms.Compose([
            transforms.Resize(target_width),
            transforms.CenterCrop(target_width),
            transforms.ToTensor(),
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
        ])
    return t(image).unsqueeze(0)


def read_image(image, target_width):
    return preprocess_image(Image.fromarray((image * 255).astype(np.uint8)).convert('RGB'), target_width)


def predict_vgg(image_array, style_array, resize_flag, itt_numb=50):
    datax, datay, _ = 256 - np.array(image_array.shape)
    offsetx1, offsetx2 = (datax//2, datax//2) if datax % 2 == 0 else (datax//2, datax//2 + 1)
    offsety1, offsety2 = (datay//2, datay//2) if datay % 2 == 0 else (datay//2, datay//2 + 1)

    stylex, styley, _ = 256 - np.array(style_array.shape)
    offsetx1_st, offsetx2_st = (stylex//2, stylex//2) if stylex % 2 == 0 else (stylex//2, stylex//2 + 1)
    offsety1_st, offsety2_st = (styley//2, styley//2) if styley % 2 == 0 else (styley//2, styley//2 + 1)
    result_list = np.zeros_like(image_array)
    target_width = None

    # result_list = np.zeros_like(image_array)
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = VGG(vgg16.features[:23]).to('cuda').eval()
    vgg16 = vgg16.to(device)

    for pos, (content, style) in enumerate(tqdm(zip(np.rollaxis(image_array, 2), np.rollaxis(style_array, 2)))):
        # style = np.stack((style,)*3, axis=2)
        if not resize_flag:
            style = np.pad(style, ((offsetx1_st, offsetx2_st), (offsety1_st, offsety2_st)), 'constant')
        style_img = read_image(style, target_width)

        style_img = style_img.to(device)

        if not resize_flag:
            content = np.pad(content, ((offsetx1, offsetx2), (offsety1, offsety2)), 'constant')
        content_img = read_image(content, target_width)
        content_img = content_img.to(device)


        style_features = vgg16(style_img)
        content_features = vgg16(content_img)

        style_grams = [gram_matrix(x) for x in style_features]

        input_img = content_img.clone()
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        style_weight = 1e6
        content_weight = 1

        run = [0]
        while run[0] <= itt_numb:
            def f():
                optimizer.zero_grad()
                features = vgg16(input_img)

                content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
                style_loss = 0
                grams = [gram_matrix(x) for x in features]
                for a, b in zip(grams, style_grams):
                    style_loss += F.mse_loss(a, b) * style_weight

                loss = style_loss + content_loss

                # if run[0] % 10 == 0:
                    # print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(
                    #     run[0], style_loss.item(), content_loss.item()))
                    # result_values.append(input_img.clone())
                run[0] += 1

                loss.backward()
                return loss

            optimizer.step(f)

        input_img = input_img.detach().cpu().numpy()
        input_img = input_img[0, 0, ...]
        if not resize_flag:
            input_img = input_img[offsetx1: 256 - offsetx2, offsety1: 256 - offsety2]
        result_list[..., pos] = input_img

    return result_list
