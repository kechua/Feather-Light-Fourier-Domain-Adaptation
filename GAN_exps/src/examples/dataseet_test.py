from typing import List

import torch
from torch.utils.data import DataLoader

from dataset.DA_dataloader import CustomPictDataset
from dataset.lazy_loader import LazyLoader
from gan.nn.stylegan.generator import FromStyleConditionalGenerator, Decoder, Generator
from gan.nn.stylegan.style_encoder import GradualStyleEncoder


# image_generator = Generator(FromStyleConditionalGenerator(256, 512, style_multiplayer=2)).cuda()
#
# decoder = Decoder(image_generator).cuda()
# style_enc = GradualStyleEncoder(50, 1, mode="ir", style_count=14, style_multiplayer=2).cuda()


batch = next(LazyLoader.domain_adaptation_siemens3().loader_train_inf)
image, mask = batch['image'], batch['mask']
print('Image shape:', image.shape)
print('Mask shape:', mask.shape)
# latent = style_enc.forward(image.cuda())
# print(latent.shape)
# reconstructed = decoder.forward([latent[:, k] for k in range(latent.shape[1])])
# print(reconstructed.shape)
# print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
# print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')
# #

# batch = next(LazyLoader.domain_adaptation_ge3().loader_train_inf)
# image, mask = batch['image'], batch['mask']
# print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
# print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')
# from gan.nn.stylegan.components import EqualLinear
# from gan.nn.stylegan.discriminator import CondBinaryDiscriminator
# from nn.progressiya.base import Progressive, ProgressiveWithoutState

# batch = next(LazyLoader.celeba().loader)
# print(batch.shape)

# batch = next(LazyLoader.metfaces().loader)
# print(batch.shape)

#
# model = Noise2Style()
# image_generation = ConditionalStyleTransform()
# print (image_generation(model.forward(batch_size=8), cond=torch.zeros(8, dtype=torch.int64).cuda()).shape)
# qwq = CondBinaryDiscriminator(size=256)
# print(CondBinaryDiscriminator(size=256).forward(input=image_generation(model.forward(batch_size=8),
#                                                                        cond=torch.zeros(8, dtype=torch.int64).cuda()),
#                                                 cond=0).shape)

#
# batch = next(LazyLoader.domain_adaptation_siemens15().loader_train_inf)
# image, mask = batch['image'], batch['mask']
# print(f'{image.shape}: image.shape, {mask.shape}: mask.shape')
# print(f'{image.min()}: image.min, {image.max()}: image.max,  {torch.unique(mask)}: mask.unique')

