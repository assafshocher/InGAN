import torch
from torch.nn import functional as f
import numpy as np


def affine_based_on_top_left_corner_x_shift(rand_affine):
    """
    random affine transformation that only shifts the top-left corner at random along the x direction
    :param sig: amount of random x perturbation
    :return: forward and backward affine transforms
    """
    aff = np.array([[1., -0.5 * rand_affine, 0.5 * rand_affine], [0, 1., 0]], dtype=np.float32)

    return torch.from_numpy(aff).clone().cuda()


def apply_resize_and_affine(x, target_size, rand_affine):
    aff = affine_based_on_top_left_corner_x_shift(rand_affine)
    target_size4d = torch.Size([x.shape[0], x.shape[1], target_size[0], target_size[1]])
    grid = f.affine_grid(aff.expand(x.shape[0], -1, -1), target_size4d)
    out = f.grid_sample(x, grid, mode='bilinear', padding_mode='border')
    return out


def homography_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of homography matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Tensor): input batch of homography matrices (:math:`N \times 3 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    """
    y, x = torch.meshgrid((torch.linspace(-1., 1., size[-2]), torch.linspace(-1., 1., size[-1])))
    n = size[-2] * size[-1]
    hxy = torch.ones(n, 3, dtype=torch.float)
    hxy[:, 0] = x.contiguous().view(-1)
    hxy[:, 1] = y.contiguous().view(-1)
    out = hxy[None, ...].cuda().matmul(theta.transpose(1, 2))
    # normalize
    out = out[:, :, :2] / out[:, :, 2:]
    return out.view(theta.shape[0], size[-2], size[-1], 2)


def apply_resize_and_homograhpy(x, target_size, rand_h):
    theta = homography_based_on_top_corners_x_shift(rand_h)
    target_size4d = torch.Size([x.shape[0], x.shape[1], target_size[0], target_size[1]])
    grid = homography_grid(theta.expand(x.shape[0], -1, -1), target_size4d)
    out = f.grid_sample(x, grid, mode='bilinear', padding_mode='border')
    return out


def homography_based_on_top_corners_x_shift(rand_h):
    # play with both top corners
    # p = np.array([[1., 1., -1, 0, 0, 0, -(-1. + rand_h[0]), -(-1. + rand_h[0]), -1. + rand_h[0]],
    #               [0, 0, 0, 1., 1., -1., 1., 1., -1.],
    #               [-1., 1., -1, 0, 0, 0, 1 + rand_h[1], -(1 + rand_h[1]), 1 + rand_h[1]],
    #               [0, 0, 0, -1, 1, -1, -1, 1, -1],
    #               [1, 0, -1, 0, 0, 0, 1, 0, -1],
    #               [0, 0, 0, 1, 0, -1, 0, 0, 0],
    #               [-1, 0, -1, 0, 0, 0, 1, 0, 1],
    #               [0, 0, 0, -1, 0, -1, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    # play with top left and bottom right
    p = np.array(
        [
            [1., 1., -1, 0, 0, 0, -(-1. + rand_h[0]), -(-1. + rand_h[0]), -1. + rand_h[0]],
            [0, 0, 0, 1., 1., -1., 1., 1., -1.], [-1., -1., -1, 0, 0, 0, 1 + rand_h[1], 1 + rand_h[1], 1 + rand_h[1]],
            [0, 0, 0, -1, -1, -1, 1, 1, 1], [1, 0, -1, 0, 0, 0, 1, 0, -1], [0, 0, 0, 1, 0, -1, 0, 0, 0],
            [-1, 0, -1, 0, 0, 0, 1, 0, 1], [0, 0, 0, -1, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ],
        dtype=np.float32
    )
    b = np.zeros((9, 1), dtype=np.float32)
    b[8, 0] = 1.
    h = np.dot(np.linalg.inv(p), b)
    return torch.from_numpy(h).view(3, 3).clone().cuda()


def apply_resize_and_radial(x, target_size, rand_r):
    target_size4d = torch.Size([x.shape[0], x.shape[1], target_size[0], target_size[1]])
    grid = make_radial_scale_grid(rand_r, target_size4d)
    out = f.grid_sample(x, grid, mode='bilinear', padding_mode='border')
    return out


def make_radial_scale_grid(rand_r, size4d):
    y, x = torch.meshgrid((torch.linspace(-1., 1., size4d[-2]), torch.linspace(-1., 1., size4d[-1])))
    theta = torch.atan2(x, y)
    r = torch.sqrt()


'''
def test_time():
    def _make_pink_noise(sz_):
        with torch.no_grad():
            n = 4  # number of scales
            pn_ = 0.
            sf = 0.375
            nsf = 0.5
            for sc in range(n):
                csz = [int(s_ * sf ** sc) for s_ in sz_[2:]]
                cn = torch.randn(sz_[0], sz_[1], csz[0], csz[1]).cuda() * nsf ** (n - sc - 1)
                pn_ += f.interpolate(cn, sz_[2:], mode='bilinear', align_corners=False)
        return torch.clamp(pn_, -1., 1.)

    import torch
    from torch.nn import functional as f
    from PIL import Image
    import util
    from InGAN import InGAN
    from configs import Config
    from skvideo.io import FFmpegWriter
    from non_rect import affine_based_on_top_left_corner_x_shift
    import numpy as np
    from non_rect import *

    conf = Config().parse()
    gan = InGAN(conf)
    sd = torch.load('results/rome_s-aff_Mar_03_16_23_22/checkpoint_0080000.pth.tar')
    gan.G.load_state_dict(sd['G'])

    def _make_affine_mask(in_mask, target_size, rand_affine):
        aff = affine_based_on_top_left_corner_x_shift(rand_affine)
        target_size4d = torch.Size([in_mask.shape[0], in_mask.shape[1], target_size[0], target_size[1]])
        grid = f.affine_grid(aff.expand(in_mask.shape[0], -1, -1), target_size4d)
        out_mask = f.grid_sample(in_mask, grid, mode='bilinear', padding_mode='zeros')
        return out_mask

    def _make_homography_mask(in_mask, target_size, rand_h):
        theta = homography_based_on_top_corners_x_shift(rand_h)
        target_size4d = torch.Size([in_mask.shape[0], in_mask.shape[1], target_size[0], target_size[1]])
        grid = homography_grid(theta.expand(in_mask.shape[0], -1, -1), target_size4d)
        out = f.grid_sample(in_mask, grid, mode='bilinear', padding_mode='zeros')
        return out

    orig = util.read_shave_tensorize('/home/bagon/develop/waic/InGAN/rome_s.png', 8)
    pad = torch.zeros(1, 3, orig.shape[2], orig.shape[3] * 2, dtype=torch.float).cuda()
    hp = orig.shape[3] // 2
    pad[..., hp:-hp] = orig
    in_mask = torch.zeros_like(pad[:, :1, ...])
    in_mask[..., hp:-hp] = 1.

    pinkn = _make_pink_noise(pad.shape)

    writer = FFmpegWriter('vid-h-fruits_ss.mp4', verbosity=1, outputdict={'-b': '30000000', '-r': '10.0'})
    n = 400
    for i in range(n):
        rand_h = (.25 * np.sin(2*np.pi*float(i)/float(0.5*n)), .25 * np.sin(2*np.pi*float(i)/float(0.25*n)))
        # a = float(.3 * np.sin(2*np.pi*float(i)/float(0.5*n)))
        out = gan.G(pad + 0. * pinkn, pad.shape[2:], rand_h)
        # out_mask = _make_affine_mask(in_mask, pad.shape[2:], a)
        out_mask = _make_homography_mask(in_mask, pad.shape[2:], rand_h)
        frame = util.tensor2im(out*out_mask - 1 + out_mask)
        writer.writeFrame(frame)
    writer.close()
'''
