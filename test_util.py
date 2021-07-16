from pytest import fixture
import torch
import numpy as np
from PIL import Image
from util import tensor2im, im2tensor


@fixture
def test_image():
    img = Image.open('examples/fruit/fruit.png')
    img = np.array(img)
    return img


def test_tensor2im(test_image):
    tensor = torch.tensor(test_image).permute(2, 0, 1).unsqueeze(0) / 255. * 2 - 1
    img = tensor2im(tensor)
    assert np.allclose(img, test_image)
