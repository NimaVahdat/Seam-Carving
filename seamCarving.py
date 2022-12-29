import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import cv2
import kornia.filters as kf


def open_image(name):
    img = cv2.imread(name)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    return img


def cal_energy(img):
    img_copy = img[None, :]

    g = kf.spatial_gradient(img_copy, mode="sobel")
    Ix = g[:, :, 0, :, :]
    Iy = g[:, :, 1, :, :]
    G = torch.hypot(Ix, Iy)
    energy = torch.squeeze(torch.sum(G, dim=1))

    return energy


def CravingHelper(img, reduce_size):
    for _ in range(reduce_size):
        E = cal_energy(img)

        M = torch.zeros((img.shape[1], img.shape[2]), dtype=torch.float32)
        M[0, :] = E[0, :]

        M = F.pad(M, (1, 1), value=float("inf"))

        for i in range(1, M.shape[0]):
            for j in range(1, M.shape[1] - 1):
                M[i, j] = E[i, j - 1] + min(M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1])

        col = torch.argmin(M[M.shape[0] - 1, :])
        path = [[M.shape[0] - 1, col]]
        for i in range(M.shape[0] - 2, 0, -1):
            values = [M[i, col - 1], M[i, col], M[i, col + 1]]
            col = col + values.index(min(values)) - 1
            path.append([i, col])

        result = torch.zeros((3, img.shape[1], img.shape[2] - 1), dtype=torch.float32)

        for i in path:
            result[:, i[0], :] = torch.concat((img[:, i[0], : i[1] - 1], img[:, i[0], i[1] :]), 1)

        img = torch.clone(result)

    return img


def MySeamCraving(img, resolution):
    W, H = resolution

    H_redu = img.shape[2] - H
    W_redu = img.shape[1] - W

    print(H_redu, W_redu)
    img = CravingHelper(img, H_redu)

    img = torch.transpose(img, 1, 2)
    img = CravingHelper(img, W_redu)

    img = torch.transpose(img, 1, 2)

    return img


if __name__ == "__main__":
    img_mine = open_image("Images/james_dean.jpeg")
    print(img_mine.shape)
    img = MySeamCraving(img_mine, (800, 650))
    plt.imshow(img.permute(1, 2, 0))
