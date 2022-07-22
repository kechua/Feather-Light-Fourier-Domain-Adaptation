import numpy as np
import torch


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    #     mask = (radius == dist_from_center) or (dist_from_center < (radius + 1))
    mask = np.logical_or(radius == dist_from_center, dist_from_center < (radius + 1))
    return mask


def create_rect_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))

    perc = radius / min(h, w) * 2

    mask = np.zeros((h, w))

    for x in range(h // 2 - int(h // 2 * perc), h // 2 + int(h // 2 * perc)):
        for y in range(w // 2 - int(w // 2 * perc), w // 2 + int(w // 2 * perc)):
            mask[x][y] = 1

    return mask.astype(bool)

def mask_applying_exp(fourier1, fourier2, rad, circle=True):


    image_amp1, image_phase1 = np.absolute(fourier1), np.angle(fourier1)
    image_amp2, image_phase2 = np.absolute(fourier2), np.angle(fourier2)

    if circle:
        mask = create_circular_mask(300, 300, radius=rad)
    else:
        mask = create_rect_mask(300, 300, radius=rad)

    image_amp1[mask] = image_amp2[mask]

    fourier_image = image_amp1 * np.exp(-image_phase1 * 1.0j)
    fourier_image = np.fft.ifftshift(fourier_image)

    return np.abs(np.fft.ifft2(fourier_image))


def replacing_with_mask_basic(image1, image2, rad):

    fourier1 = np.fft.fftshift(np.fft.fft2(np.asarray(image1)))
    fourier2 = np.fft.fftshift(np.fft.fft2(np.asarray(image2)))

    return mask_applying(fourier1, fourier2, rad)


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    if n == 0:
        n = 1
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def create_mask(divers, perc, mask_size):
    if perc > 0.5:
        print('It is a complete transfer. Check the percentage parameter again.')

    amount = int(mask_size * perc)
    mask = np.zeros_like(divers)
    mask[largest_indices(divers, amount)] = 1
    mask = mask.astype(bool)
    return mask


def mask_applying(image1, image2, mask, mask_convering=False, with_shift=True, phase=False):
    fourier1 = torch.fft.fft2(torch.squeeze(image1))
    if with_shift:
        fourier1 = torch.fft.fftshift(fourier1)
    image_amp1, image_phase1 = torch.abs(fourier1), torch.angle(fourier1)
    # image_amp1, image_phase1 = np.absolute(fourier1), np.angle(fourier1)

    if not mask_convering:
        fourier2 = torch.fft.fft2(image2)
        if with_shift:
            fourier2 = torch.fft.fftshift(fourier2)

        image_amp2, image_phase2 = torch.abs(fourier2), torch.angle(fourier2)
    else:
        image_amp2, image_phase2 = image2, image2

    if phase:
        phase = image_phase1.copy()
        phase[mask == 1] = image_phase2[mask]
        amp = image_amp1
    else:
        amp = image_amp1
        image_amp2 = torch.from_numpy(image_amp2).float().to('cuda').unsqueeze(0)
        # mask = mask[None, ...]

        mask = torch.from_numpy(mask).to('cuda').unsqueeze(0)
        image_amp2 = image_amp2.expand(8,300,300)
        mask = mask.expand(8,300,300)

        amp[mask == 1] = image_amp2[mask]
        # amp = image_amp2
        phase = image_phase1

    fourier_image = amp * torch.exp(-phase * 1.0j)

    if with_shift:
        fourier_image = torch.fft.ifftshift(fourier_image)

    return torch.abs(torch.fft.ifft2(fourier_image))

def get_image_sum(image, values_amp, values_phase):
    fourier = np.fft.fft2(np.asarray(image))
    fourier = np.fft.fftshift(fourier)

    image_amp, image_phase = np.absolute(fourier), np.angle(fourier)
    values_amp += image_amp
    values_phase += image_phase
    return values_amp, values_phase
