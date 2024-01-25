#!/usr/bin/env python3

from typing import Tuple
import torch as th
from icecream import ic


# def box_from_mask(mask: th.Tensor):
#     ii, jj = th.nonzero(mask, as_tuple=True)
#     i0, j0 = ii.min(), jj.min()
#     i1, j1 = ii.max() + 1, jj.max() + 1
#     return (i0, j0, i1, j1)

def box_from_mask(img):
    """
    img: boolean array(..., h, w)
    From:
    https://discuss.pytorch.org/t/find-bounding-box-around-ones-in-batch-of-masks/141266/2
    """

    # TODO(ycho): speed up this routine...!
    h, w = img.shape[-2:]
    ii = th.any(img, axis=-1).float()
    jj = th.any(img, axis=-2).float()
    rmins = th.argmax(ii, dim=1)
    rmaxs = h - th.argmax(ii.flip(dims=[-1]), dim=1)
    cmins = th.argmax(jj, dim=1)
    cmaxs = w - th.argmax(jj.flip(dims=[-1]), dim=1)
    return rmins, cmins, rmaxs, cmaxs


def crop_params_from_mask(mask: th.Tensor,
                          shape: Tuple[int, int],
                          margin: float = 0.1,
                          # square: bool = True,
                          # match
                          match_aspect:bool=True
                          ):
    """
    Find the bounding crop in an image,
    assuming it finds where there is non-zero.
    Args:
        mask: Input image tensor: (...HW)
    Returns:
        list: [i0, j0, size]
    """

    # Find the bounding box of the img_tensor
    i0, j0, i1, j1 = box_from_mask(mask)
    # ic(i0, j0,
    #    i1, j1)

    # Calculate the wiggle room for each dimension (percentage of the
    # width/height)
    mi = ((i1 - i0) * margin)
    mj = ((j1 - j0) * margin)

    # Expand the bounding box by the wiggle room
    i0 = (i0 - mi)#.clamp(min=0)
    j0 = (j0 - mj)#.clamp(min=0)
    i1 = (i1 + mi)#.clamp(max=mask.shape[-2])
    j1 = (j1 + mj)#.clamp(max=mask.shape[-1])

    di, dj = (i1 - i0, j1 - j0)

    # Try to achieve
    # isometric scaling i.e. scale[0] == scale[1]
    if match_aspect:
        # di * s = shape[0]
        # dj * s = shape[1]
        s = th.maximum(di/shape[0],
                       dj/shape[1])
        di = shape[0] * s
        dj = shape[1] * s
        # s = shape[0]/di = shape[1]/dj

        # scale = max(shape) / th.maximum(di, dj)
        # di *= scale
        # dj *= scale
        # mx = th.maximum(di, dj)
        # di, dj = mx, mx

    scale = (shape[0] / di, shape[1] / dj)
    return [i0, j0, di, dj], scale


def flip_crop(shape: Tuple[int, int],
              crop: Tuple[int, int, int, int]):
    h = shape[0]
    crop = [h - crop[0] - crop[2],
            crop[1],
            crop[2],
            crop[3]]


def crop_and_scale_intrinsics(
        fx: th.Tensor,
        fy: th.Tensor,
        cx: th.Tensor,
        cy: th.Tensor,
        crop: Tuple[int, int, int, int],
        scale: Tuple[float, float]):
    #ic(fx,
    #   fy,
    #   cx,
    #   cy,
    #   crop,
    #   scale)
    sy, sx = scale
    y0, x0, _, _ = crop

    return (
        fx * sx,
        fy * sy,
        (cx - x0) * sx,
        (cy - y0) * sy
    )


def ndc_projection(
        # Image dimensions.
        # also works for cropped
        w: th.Tensor,
        h: th.Tensor,

        # Camera intrinsic parameters;
        # we don't consider distortions or
        # "tilt" angles.
        fx: th.Tensor,
        fy: th.Tensor,
        cx: th.Tensor,
        cy: th.Tensor,

        # Frustum clipping parameters;
        # I guess these can be tensors too
        z_n: float,
        z_f: float,
):
    """
    Adapted from diff-dope
    """
    # ic(w, h,
    #    fx, fy, cx, cy,
    #    z_n, z_f)
    z_d = (z_f - z_n)
    q = -(z_f + z_n) / z_d
    qn = -2 * (z_f * z_n) / z_d

    P = th.zeros((fx.shape + (4, 4)),
                 dtype=fx.dtype,
                 device=fx.device)
    P[..., 0, 0] = 2 * fx / w
    P[..., 0, 2] = (w - 2 * cx) / w
    P[..., 1, 1] = 2 * fy / h
    P[..., 1, 2] = (2 * cy - h) / h
    P[..., 2, 2] = q
    P[..., 2, 3] = qn
    P[..., 3, 2] = -1
    return P
