import logging

import matplotlib

matplotlib.use("Agg")

import collections
import io
import sys
import math
import pathlib
import random
import warnings
from dataclasses import dataclass
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from typing import Dict

import cv2
import hydra
import hydra.utils as hydra_utils
import imageio
import matplotlib.pyplot as plt
import numpy as np
import nvdiffrast.torch as dr
import pyrr
import torch
import torch as th
import torch.nn as nn
import einops
import trimesh
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from PIL import Image as pilImage
from PIL import ImageColor, ImageDraw, ImageFont
from tqdm import tqdm
import nvtx

import opt_einsum as oe

import diffdope as dd
from pkm.util.torch_util import dcn
from pkm.util.math_util import (
    r6_from_quat,
    matrix_from_r6,
    axa_from_quat,
    matrix_from_quaternion
)
from torchviz import (make_dot, make_dot_from_trace)

from diffdope.lie_utils import SE3Exp, SE3

# for better print debug
print()
if not hasattr(sys, 'ps1'):
    print = ic

# A logger for this file
log = logging.getLogger(__name__)

# adjust rotation parameterization
RTYPE = 'se3'
# adjust rotation parameter 'sensitivity'
# higher scale = more sensitive (i.e. more likely to change)
SCALE = 10.0
S_TEX = 0.5
LOCAL: bool = True


def opencv_2_opengl(p, q):
    """
    Converts a pose from the OpenCV coordinate system to the OpenGL coordinate system.

    Args:
        p (np.ndarray): position
        q (pyrr.Quaternion): quat

    Returns:
        p,q
    """
    # source_transform = q.matrix44
    # source_transform[:3, 3] = p
    print(pyrr.Matrix44.from_translation(p))
    source_transform = q.matrix44 @ pyrr.Matrix44.from_translation(p)
    opengl_to_opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    R_opengl_to_opencv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    t_opengl_to_opencv = np.array([0, 0, 0])

    # Adjust rotation and translation for target coordinate system
    adjusted_rotation = np.dot(R_opengl_to_opencv, source_transform[:3, :3])
    adjusted_translation = (
        np.dot(R_opengl_to_opencv, source_transform[3, : 3]) +
        t_opengl_to_opencv)
    # print(p)
    # print(source_transform[:3, 3])
    # print(np.dot(R_opengl_to_opencv, source_transform[:3, 3]))
    # print(R_opengl_to_opencv)
    # print(adjusted_translation)

    # Build target transformation matrix (OpenCV convention)
    target_transform = np.eye(4)
    target_transform[:3, :3] = adjusted_rotation
    target_transform[3, :3] = adjusted_translation

    q = pyrr.Matrix44(target_transform).quaternion

    # TODO verify what is going on here, this should not be needed ...
    # legacy code here
    q = (
        q
        * pyrr.Quaternion.from_z_rotation(np.pi / 2)
        * pyrr.Quaternion.from_y_rotation(-np.pi / 2)
    )
    q = (
        q
        * pyrr.Quaternion.from_z_rotation(-np.pi / 2)
        * pyrr.Quaternion.from_x_rotation(-np.pi / 2)
    )
    # END TODO
    print(target_transform[3, :3])

    return target_transform[3, :3], q


def interpolate(attr, rast, attr_idx, rast_db=None):
    """
    A wrapper around nvdiffrast interpolate
    """
    return dr.interpolate(
        attr.contiguous(),
        rast,
        attr_idx,
        rast_db=rast_db,
        diff_attrs=None if rast_db is None else "all",
    )


def render_texture_batch(
    glctx,
    proj_cam,
    mtx,
    posw,
    pos_idx,
    resolution,
    uv=None,
    uv_idx=None,
    tex=None,
    vtx_color=None,
    return_rast_out=False,
    oe_expr=None
):
    """
    Functions that maps 3d objects to the nvdiffrast rendering. This function uses the color of the texture of the model or the vertex color.
    If there is no illumination on the object model then it will look flat, we recommend to bake in some lights, see blender script provided.
    See the mesh class if you want more information about how to construct your mesh for rendering.

    Args:
    glctx (): the nvdiffrast context
    proj_cam (torch.tensor): (b,4,4) is the camera projection matrix
    mtx (torch.tensor): (b,4,4) the world camera pose express in opengl coordinate system
    pos (torch.tensor): (b,nb_points,3) the object 3d points defining the 3d model
    pos_idx (torch.tensor): (nb_points,3) the object triangle list
    resolution (np.ndarray): (2) the image resolution to be render
    uv (torch.tensor): (b,nb_points,2) where each object point lands on the texture
    uv_idx (torch.tensor): (b,nb_points,3) defining each texture triangle
    tex (torch.tensor): (b,w,h,3) batch image of the texture
    vtx_color (torch.tensor): (b,nb_points,3) the color of each vertex, this is used when the texture is not defined
    return_rast_out (bool): return the nvdiffrast output

    Return:
        returns a dict with key 'rgb','depth', and 'rast_out'
    """
    if not type(resolution) == list:
        resolution = [resolution, resolution]

    # Potentially better to keep it
    # as einsum() to exploit torch.compile()
    pos_clip_ja = th.einsum('nij, njk, n...k -> n...i',
                            proj_cam, mtx, posw)
    # pos_clip_ja = oe_expr(proj_cam, mtx, posw)
    # pos_clip_ja = oe_expr(mtx).contiguous()

    rast_out, rast_out_db = dr.rasterize(
        glctx, pos_clip_ja, pos_idx[0], resolution=resolution
    )

    # TODO(ycho): optionally include depth losses
    depth = None
    if False:
        gb_pos, _ = interpolate(
            posw, rast_out, pos_idx[0],
            rast_db=rast_out_db)
        shape_keep = gb_pos.shape
        gb_pos = gb_pos.reshape(shape_keep[0], -1, shape_keep[-1])
        gb_pos[..., 3].fill_(1)
        depth = th.matmul(gb_pos, mtx[..., 2, :, None]).squeeze(dim=-1)
        depth = -depth.reshape(shape_keep[:-1])

    # TODO(ycho): optionally include mask losses
    mask = None
    if False:
        mask, _ = dr.interpolate(
            # torch.ones(pos_idx.shape).cuda(),
            th.ones(pos_idx.shape, device='cuda'),
            rast_out, pos_idx[0], rast_db=rast_out_db, diff_attrs="all")
        mask = dr.antialias(mask, rast_out, pos_clip_ja, pos_idx[0])

    # compute vertex color interpolation
    if vtx_color is None:
        texc, texd = dr.interpolate(
            uv, rast_out, uv_idx[0], rast_db=rast_out_db, diff_attrs="all"
        )
        color = dr.texture(
            tex,
            texc,
            texd,
            filter_mode="linear",
        )
    else:
        color, _ = dr.interpolate(vtx_color, rast_out, pos_idx[0])
    return {"rgb": color, "depth": depth, "rast_out": rast_out, 'mask': mask}

@torch.compile()
def loss_fn(
        gt_tensors:Dict[str, th.Tensor],
        proj_cam:th.Tensor,
        #mtx_gu:th.Tensor,

        T0:th.Tensor,
        u_se3:th.Tensor,
        scale:th.Tensor,
        g:th.Tensor,

        posw:th.Tensor,
        pos_idx:th.Tensor,
        uv:th.Tensor,
        uv_idx:th.Tensor,
        tex:th.Tensor,
        learning_rates:th.Tensor,
        weight_rgb:float,
        resolution:Tuple[int,int],
        # oe_expr,
        glctx
    ):
    mtx_gu = T0 @ SE3Exp(u_se3 * scale, g)
    renders = render_texture_batch(
                            glctx=glctx,
                            proj_cam=proj_cam,
                            mtx=mtx_gu,
                            posw=posw,
                            pos_idx=pos_idx,
                            uv=uv,
                            uv_idx=uv_idx,
                            tex=tex,
                            resolution=resolution)
    return l1_rgb_with_mask_t(
                # ddope.renders['rgb'][..., :3],
                renders['rgb'],
                gt_tensors['rgb'][..., :3],
                renders['rast_out'][..., -1:],
                gt_tensors["segmentation"],
                learning_rates,
                weight_rgb), mtx_gu, renders['rgb']

##############################################################################
# LOSSES
##############################################################################
def l1_rgb_with_mask_t(pred_rgb:th.Tensor,
                       true_rgb:th.Tensor,
                       pred_mask: th.Tensor,
                       true_mask:th.Tensor,
                       learning_rates:th.Tensor,
                       weight_rgb:float) -> th.Tensor:
    diff_rgb = torch.abs((pred_rgb * (pred_mask>0) - true_rgb) * (true_mask))
    # lr_diff_rgb = dist_batch_lr(diff_rgb, learning_rates)
    batch_err = diff_rgb.reshape(diff_rgb.shape[0], -1).mean(dim=-1)
    lr_diff_rgb = batch_err * learning_rates
    return lr_diff_rgb.mean() * weight_rgb, batch_err


def l1_rgb_with_mask(ddope):
    """
    Computes the l1_rgb on a DiffDOPE object, simpler to pass the object.
    """
    return l1_rgb_with_mask_t(
            # ddope.renders['rgb'][..., :3],
            ddope.renders['rgb'],
            ddope.gt_tensors['rgb'][..., :3],
            ddope.renders['rast_out'][..., -1:],
            ddope.gt_tensors["segmentation"],
            ddope.learning_rates,
            ddope.cfg.losses.weight_rgb)


def l1_depth_with_mask(ddope):
    """
    Computes the l1_depth on a DiffDOPE object, simpler to pass the object.
    """

    diff_depth = torch.abs(
        (ddope.renders["depth"] - ddope.gt_tensors["depth"])
        * ddope.gt_tensors["segmentation"][..., 0]
    )
    lr_diff_depth = dist_batch_lr(diff_depth, ddope.learning_rates, [1, 2])

    ddope.add_loss_value(
        "depth", torch.mean(
            diff_depth.detach(), (1, 2)) * ddope.cfg.losses.weight_depth)

    return lr_diff_depth.mean() * ddope.cfg.losses.weight_depth


def l1_mask(ddope):
    """
    Computes the L1-on mask on a DiffDOPE object.

    Args:
    - ddope: A DiffDOPE object containing the necessary data.

    Returns:
    - lr_diff_mask_mean: The mean of the L1-on mask loss multiplied by the weight specified in the configuration.
    """

    mask = ddope.renders["mask"]
    ddope.optimization_results[-1]["mask"] = mask.detach()  # .cpu()

    # Compute the difference between the mask and ground truth segmentation
    diff_mask = torch.abs(mask - ddope.gt_tensors["segmentation"])

    # Compute the L1-on mask loss with batch-wise learning rates
    lr_diff_mask = dist_batch_lr(diff_mask, ddope.learning_rates)

    # Add the L1-on mask loss to the DiffDOPE object
    ddope.add_loss_value(
        "mask_selection",
        torch.mean(torch.abs(diff_mask.detach()), (1, 2, 3))
        * ddope.cfg.losses.weight_mask,
    )

    # Calculate the mean of the L1-on mask loss and apply the weight from the
    # configuration
    lr_diff_mask_mean = lr_diff_mask.mean() * ddope.cfg.losses.weight_mask

    return lr_diff_mask_mean


##############################################################################
# CLASSES
##############################################################################


@dataclass
class Camera:
    """
    A class for representing the camera, mostly to store classic computer vision oriented reprojection values
    and then get the OpenGL projection matrix out.

    Args:
        fx (float): focal length x-axis in pixel unit
        fy (float): focal length y-axis in pixel unit
        cx (float): principal point x-axis in pixel
        cy (float): principal point y-axis in pixel
        im_width (int): width of the image
        im_height (int): height of the image
        znear (float, optional): for the opengl reprojection, how close can a point be to the camera before it is clipped
        zfar (float, optional): for the opengl reprojection, how far can a point be to the camera before it is clipped
    """

    fx: float
    fy: float
    cx: float
    cy: float
    im_width: int
    im_height: int
    znear: Optional[float] = 0.01
    zfar: Optional[float] = 200

    def __post_init__(self):
        self.cam_proj = self.get_projection_matrix()

    def set_batchsize(self, batchsize):
        """
        Change the batchsize for the image tensor

        Args:
            batchsize (int): batchsize for the tensor
        """
        if len(self.cam_proj.shape) == 2:
            self.cam_proj = torch.stack([self.cam_proj] * batchsize, dim=0)
        else:
            self.cam_proj = torch.stack([self.cam_proj[0]] * batchsize, dim=0)

    def cuda(self):
        self.cam_proj = self.cam_proj.cuda().float()

    def resize(self, percentage):
        """
        If you resize the images for the optimization

        Args:
            percentage (float): bounded between [0,1]
        """
        self.fx *= percentage
        self.fy *= percentage
        self.cx = (int)(percentage * self.cx)
        self.cy = (int)(percentage * self.cy)
        self.im_width = (int)(percentage * self.im_width)
        self.im_height = (int)(percentage * self.im_height)

    def get_projection_matrix(self):
        """
        Conversion of Hartley-Zisserman intrinsic matrix to OpenGL projection matrix.

        Refs:

        1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
        2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
        3) https://github.com/megapose6d/megapose6d/blob/3f5b51d9cef71d9ac0ac36c6414f35013bee2b0b/src/megapose/panda3d_renderer/types.py

        Returns:
            torch.tensor: a 4x4 projection matrix in OpenGL coordinate frame
        """

        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        x0 = 0
        y0 = 0
        w = self.im_width
        h = self.im_height
        nc = self.znear
        fc = self.zfar

        window_coords = "y_down"

        depth = float(fc - nc)
        q = -(fc + nc) / depth
        qn = -2 * (fc * nc) / depth

        # Draw our images upside down, so that all the pixel-based coordinate
        # systems are the same.
        if window_coords == "y_up":
            proj = np.array(
                [
                    [
                        2 * K[0, 0] / w,
                        -2 * K[0, 1] / w,
                        (-2 * K[0, 2] + w + 2 * x0) / w,
                        0,
                    ],
                    [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                    [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                    [0, 0, -1, 0],
                ]
            )

        # Draw the images upright and modify the projection matrix so that OpenGL
        # will generate window coords that compensate for the flipped image
        # coords.
        else:
            assert window_coords == "y_down"
            proj = np.array(
                [
                    [
                        2 * K[0, 0] / w,
                        -2 * K[0, 1] / w,
                        (-2 * K[0, 2] + w + 2 * x0) / w,
                        0,
                    ],
                    [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                    [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                    [0, 0, -1, 0],
                ]
            )

        return torch.tensor(proj)


# @dataclass
class Mesh(torch.nn.Module):
    """
    A wrapper around a Trimesh mesh, where the data is already loaded
    to be consumed by PyTorch. As such the internal values are stored as
    torch array.

    Args:
        path_model (str): path to the object to be loaded, see Trimesh which extensions are supported.
        scale (int): scale of mesh

    Attributes:
        pos_idx (torch.tensor): (nb_triangle,3) triangle list for the mesh
        pos (torch.tensor): (n,3) vertex positions in object space
        vtx_color (torch.tensor): (n,3) vertex color - might not exists if the file does not have that information stored
        tex (torch.tensor): (w,h,3) textured saved - might not exists if the file does not have texture
        uv (torch.tensor): (n,2) vertex uv position - might not exists if the file does not have texture
        uv_idx (torch.tensor): (nb_triangle,3) triangles for the uvs - might not exists if the file does not have texture
        bounding_volume (np.array): (2,3) minimum x,y,z with maximum x,y,z
        dimensions (list): size in all three axes
        center_point (list): position of the center of the object
        textured_map (boolean): was there a texture loaded
    """

    def __init__(self, path_model, scale):
        super().__init__()

        # load the mesh
        self.path_model = path_model
        self.to_process = [
            # "pos",
            "posw",
            "pos_idx",
            "vtx_color",
            "tex",
            "uv",
            "uv_idx",
            "vtx_normals",
        ]

        mesh = trimesh.load(self.path_model, force="mesh")

        pos = np.asarray(mesh.vertices)
        pos_idx = np.asarray(mesh.faces)

        normals = np.asarray(mesh.vertex_normals)

        pos_idx = torch.from_numpy(pos_idx.astype(np.int32))

        vtx_pos = torch.from_numpy(pos.astype(np.float32)) * scale
        vtx_normals = torch.from_numpy(normals.astype(np.float32))
        bounding_volume = [
            [
                torch.min(vtx_pos[:, 0]),
                torch.min(vtx_pos[:, 1]),
                torch.min(vtx_pos[:, 2]),
            ],
            [
                torch.max(vtx_pos[:, 0]),
                torch.max(vtx_pos[:, 1]),
                torch.max(vtx_pos[:, 2]),
            ],
        ]

        dimensions = [
            bounding_volume[1][0] - bounding_volume[0][0],
            bounding_volume[1][1] - bounding_volume[0][1],
            bounding_volume[1][2] - bounding_volume[0][2],
        ]
        center_point = [
            ((bounding_volume[0][0] + bounding_volume[1][0]) / 2).item(),
            ((bounding_volume[0][1] + bounding_volume[1][1]) / 2).item(),
            ((bounding_volume[0][2] + bounding_volume[1][2]) / 2).item(),
        ]

        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            tex = np.array(mesh.visual.material.image) / 255.0
            uv = mesh.visual.uv
            uv[:, 1] = 1 - uv[:, 1]
            uv_idx = np.asarray(mesh.faces)

            if True:
                #print(uv.m
                tex = cv2.resize(tex, dsize=None, fx=S_TEX, fy=S_TEX)
                #uv *= S_TEX

            tex = torch.from_numpy(tex.astype(np.float32))[..., :3]
            # tex = torch.from_numpy(tex.astype(np.float32))
            uv_idx = torch.from_numpy(uv_idx.astype(np.int32))
            vtx_uv = torch.from_numpy(uv.astype(np.float32))

            self.pos_idx = pos_idx
            # self.pos = vtx_pos
            self.posw = nn.functional.pad(vtx_pos,
                                         (0, 1), mode='constant', value=1)
            self.tex = tex
            self.uv = vtx_uv
            self.uv_idx = uv_idx
            self.bounding_volume = bounding_volume
            self.dimensions = dimensions
            self.center_point = center_point
            self.vtx_normals = vtx_normals
            self.has_textured_map = True

        else:
            vertex_color = mesh.visual.vertex_colors[..., :3] / 255.0
            vertex_color = torch.from_numpy(vertex_color.astype(np.float32))

            self.pos_idx = pos_idx
            # self.pos = vtx_pos
            self.posw = nn.functional.pad(vtx_pos,
                                         (0, 1), mode='constant', value=1)
            self.vtx_color = vertex_color
            self.bounding_volume = bounding_volume
            self.dimensions = dimensions
            self.center_point = center_point
            self.vtx_normals = vtx_normals
            self.has_textured_map = False

        log.info(
            f"loaded mesh @{self.path_model}. Does it have texture map? {self.has_textured_map} "
        )
        self._batchsize_set = False

    @property
    def device(self):
        return self.posw.device
    

    def __str__(self):
        return f"mesh @{self.path_model}. vtx:{self.posw.shape} on {self.posw.device}"

    def __repr__(self):
        return f"mesh @{self.path_model}. vtx:{self.posw.shape} on {self.posw.device}"

    def set_batchsize(self, batchsize):
        """
        Set the batchsize of the mesh object to match the optimization.

        Args:
            batchsize (int): batchsize for the arrays used by nv diff rast

        """

        # TODO tex does not seem to be correct in its shape

        for key, value in vars(self).items():
            if not key in self.to_process:
                continue
            if self._batchsize_set is False:
                vars(self)[key] = torch.stack(
                    [vars(self)[key]] * batchsize, dim=0)
            else:
                vars(self)[key] = torch.stack(
                    [vars(self)[key][0]] * batchsize, dim=0)

        for key, value in self._parameters.items():
            if not key in self.to_process:
                continue
            if self._batchsize_set is False:
                self._parameters[key] = torch.stack(
                    [self._parameters[key]] * batchsize, dim=0
                )
            else:
                self._parameters[key] = torch.stack(
                    [self._parameters[key][0]] * batchsize, dim=0
                )
        self.posw = self.posw.contiguous()

        if self._batchsize_set is False:
            self._batchsize_set = True

    def cuda(self):
        """
        put the arrays from `to_process` on gpu
        """
        super().cuda()

        for key, value in vars(self).items():
            if not key in self.to_process:
                continue
            vars(self)[key] = vars(self)[key].cuda()

    def enable_gradients_texture(self):
        """
        Function to enable gradients on the texture *please note* if `set_batchsize` is called after this function the gradients are set to false for the image automatically
        """
        if self.has_textured_map:
            self.tex = torch.nn.Parameter(self.tex, requires_grad=True).to(
                self.tex.device
            )
        else:
            self.vtx_color = torch.nn.Parameter(
                self.vtx_color, requires_grad=True).to(
                self.vtx_color.device)

    def forward(self):
        """
        Pass the information from the mesh back to diff-dope defined in the the `to_process`
        """
        to_return = {}
        for key, value in vars(self).items():
            if not key in self.to_process:
                continue
            to_return[key] = vars(self)[key]
        # if not 'tex' in to_return:
        #     to_return['tex'] = self.tex
        # elif not 'vtx_color' in to_return:
        #     to_return['vtx_color'] = self.vtx_color
        return to_return


class Object3D(torch.nn.Module):
    """
    This is the 3D object we want to run optimization on representation that Diff-DOPE uses to optimize.

    Attributes:
        qx,qy,qz,qw (torch.nn.Parameter): batchsize x 1 representing the quaternion
        x,y,z (torch.nn.Parameter): batchsize x 1 representing the position
        mesh (Mesh): batchsize x width x height representing the object texture, can be use in the optimization

    """

    def __init__(
        self,
        position: list,
        rotation: list,
        batchsize: int = 32,
        opencv2opengl: bool = True,
        model_path: str = None,
        scale: int = 1,
        device:str = 'cuda'
    ):
        """
        Args:
            position (list): a 3 value list of the object position
            rotation (list): could be a quat with 4 values (x,y,z,w), or a flatten rotational matrix or a 3x3 matrix (as a list of list) -- both are row-wise / row-major.
            batchsize (int): size of the batch to be optimized, this is defined normally as a hyperparameter.
            opencv2opengl (bool): converting the coordinate space from one to the other
            scale (int): scale to apply to the position
        """
        super().__init__()
        self.qx = None  # to load on cpu and not gpu

        if model_path is None:
            self.mesh = None
        else:
            self.mesh = Mesh(path_model=model_path, scale=scale)

        self.set_pose(
            position,
            rotation,
            batchsize,
            scale=scale,
            opencv2opengl=opencv2opengl,
            device=device)

    def set_pose(
        self,
        position: list,
        rotation: list,
        batchsize: int = 32,
        opencv2opengl: bool = True,
        scale: int = 1,
        device:str = None
    ):
        """
        Set the pose to new values, the inputs can be either list, numpy or torch.tensor. If the class was put on cuda(), the updated pose should be on the GPU as well.

        Args:
            position (list): a 3 value list of the object position
            rotation (list): could be a quat with 4 values (x,y,z,w), or a flatten rotational matrix or a 3x3 matrix (as a list of list) -- both are row-wise / row-major.
            batchsize (int): size of the batch to be optimized, this is defined normally as a hyperparameter.
            scale (int): scale to apply to the position

        """

        assert len(position) == 3
        position = np.array(position) * scale

        assert len(rotation) == 4 or len(rotation) == 3 or len(rotation) == 9
        if len(rotation) == 4:
            rotation = pyrr.Quaternion(rotation)
        if len(rotation) == 3 or len(rotation) == 9:
            rotation = pyrr.Matrix33(rotation).quaternion

        if opencv2opengl:
            print(position, rotation)
            position, rotation = opencv_2_opengl(position, rotation)
            print(position, rotation)

        log.info(f"translation loaded: {position}")
        log.info(f"rotation loaded as quaternion: {rotation}")

        self._position = position
        self._rotation = rotation

        # if self.qx is None:
        #     device = "cpu"
        # else:
        #     device = self.qx.device

        # self.qx = torch.nn.Parameter(torch.ones(batchsize) * rotation[0])
        # self.qy = torch.nn.Parameter(torch.ones(batchsize) * rotation[1])
        # self.qz = torch.nn.Parameter(torch.ones(batchsize) * rotation[2])
        # self.qw = torch.nn.Parameter(torch.ones(batchsize) * rotation[3])
        # self.x = torch.nn.Parameter(torch.ones(batchsize) * position[0])
        # self.y = torch.nn.Parameter(torch.ones(batchsize) * position[1])
        # self.z = torch.nn.Parameter(torch.ones(batchsize) * position[2])

        self.to(device)
        if not self.mesh is None:
            self.mesh.to(device)

    @property
    def device(self):
        # return next(self.parameters()).device
        return self.mesh.device

    def set_batchsize(self, batchsize: int):
        """
        Change the batchsize to a new value, use the latest position and rotation to reset the batch of poses with. Be careful to make sure the image data is also updated accordingly.

        Args:
            batchsize (int): Batchsize to optimize
        """
        # device = self.qx.device
        device = self.device
        # r6 = r6_from_quat(self._rotation)

        quat = th.as_tensor(np.stack([
            self._rotation[0],
            self._rotation[1],
            self._rotation[2],
            self._rotation[3]], axis=-1),
            dtype=th.float32)
        if RTYPE == 'r6':
            r6 = r6_from_quat(quat)
            br6 = einops.repeat(r6, '... -> b ...', b=batchsize)
            self.register_parameter(
                'r6', th.nn.Parameter(
                    br6.clone(), requires_grad=True))
            with th.no_grad():
                # for testing... (or for testing with perturbations...)
                # self.r6 += 0.3 * th.randn_like(self.r6)
                self.r6 /= SCALE
        elif RTYPE == 'quat':
            self.qx = torch.nn.Parameter(
                torch.ones(batchsize) * self._rotation[0])
            self.qy = torch.nn.Parameter(
                torch.ones(batchsize) * self._rotation[1])
            self.qz = torch.nn.Parameter(
                torch.ones(batchsize) * self._rotation[2])
            self.qw = torch.nn.Parameter(
                torch.ones(batchsize) * self._rotation[3])
        elif RTYPE == 'd_axa':
            self.register_buffer('q0', quat, False)
            self.d_axa = self.register_parmeter(
                'd_axa',
                nn.Parameter(torch.zeros((batchsize, 3),
                                         dtype=th.float32)))
        elif RTYPE == 'se3':
            self.register_buffer('scale',
                                 th.as_tensor([
                                     SCALE,SCALE,SCALE,
                                     1.0,1.0,1.0],
                                              dtype=th.float32),
                                 persistent=False)
            if LOCAL:
                T0 = th.zeros((batchsize, 4, 4),
                              dtype=th.float32)
                self.register_buffer('T0', T0, persistent=False)
                with th.no_grad():
                    # rotation
                    self.T0[..., :3, :3] = matrix_from_quaternion(quat)
                    # translation
                    self.T0[..., 0, 3] = self._position[0]
                    self.T0[..., 1, 3] = self._position[1]
                    self.T0[..., 2, 3] = self._position[2]
                    # homogeneous
                    self.T0[..., 3, 3] = 1
            if LOCAL:
                print('make se3')
                self.register_parameter(
                    'se3', nn.Parameter(
                        torch.zeros((batchsize, 6),
                                    dtype=th.float32)))

                with th.no_grad():
                    self.se3[..., 0] = 0
                    self.se3[..., 1] = 0
                    self.se3[..., 2] = 0
                    self.se3[..., 3] = 0
                    self.se3[..., 4] = 0
                    self.se3[..., 5] = 0
            else:
                # global
                self.register_parameter(
                    'se3', nn.Parameter(
                        torch.zeros((batchsize, 6)),
                        dtype=th.float32))
                axa = axa_from_quat(quat)
                assert (axa.shape[-1] == 3)
                with th.no_grad():
                    self.se3[..., 0] = self._position[0]
                    self.se3[..., 1] = self._position[1]
                    self.se3[..., 2] = self._position[2]
                    self.se3[..., 4] = axa[..., 0]
                    self.se3[..., 5] = axa[..., 1]
                    self.se3[..., 6] = axa[..., 2]
        for i in range(6):
            print(SE3.genmat()[i])

        self.register_buffer('g', SE3.genmat(),
                             persistent=False)
        self._oe_expr = oe.contract_expression(
                # subscripts;
                'njk,ijl,nlk -> ni',
                # constants;
                # shapes;
                (batchsize,4,4),
                # (6,4,4),
                self.g,
                (batchsize,4,4),
                constants=[1],
                optimize='optimal')

        # self.r0 = torch.nn.Parameter(torch.ones(batchsize) * r6[0])
        # self.r1 = torch.nn.Parameter(torch.ones(batchsize) * r6[1])
        # self.r2 = torch.nn.Parameter(torch.ones(batchsize) * r6[2])
        # self.r3 = torch.nn.Parameter(torch.ones(batchsize) * r6[3])
        # self.r4 = torch.nn.Parameter(torch.ones(batchsize) * r6[4])
        # self.r5 = torch.nn.Parameter(torch.ones(batchsize) * r6[5])
        # self.r6

        # self.x = torch.nn.Parameter(torch.ones(batchsize) * self._position[0])
        # self.y = torch.nn.Parameter(torch.ones(batchsize) * self._position[1])
        # self.z = torch.nn.Parameter(torch.ones(batchsize) * self._position[2])

        # p3 = th.as_tensor(np.stack([
        #         self._position[0],
        #         self._position[1],
        #         self._position[2]], axis=-1),
        #                   dtype=th.float32)
        # bp3 = einops.repeat(p3, '... -> b ...', b = batchsize)
        # self.register_parameter('pos',
        #                         th.nn.Parameter(bp3.clone(),
        #                                         requires_grad=True))

        self.to(device)

        if not self.mesh is None:
            self.mesh.set_batchsize(batchsize=batchsize)
            self.mesh.cuda()

    # def __repr__(self):
    #     # TODO use the function for the argmax
    # return f"Object3D( \n (pos): {self.x.shape}
    # ,[0]:[{self.x[0].item(),self.y[0].item(),self.z[0].item()}] on
    # {self.x.device}\n (mesh): {self.mesh} on {self.mesh.pos.device} \n)"

    def to(self, *args, **kwds):
        super().to(*args, **kwds)
        self.mesh.to(*args, **kwds)
        if hasattr(self, 'se3'):
            self._oe_expr = oe.contract_expression(
                    # subscripts;
                    'njk,ijl,nlk -> ni',
                    # constants;
                    # shapes;
                    (self.se3.shape[0],4,4),
                    # (6,4,4),
                    self.g,
                    (self.se3.shape[0],4,4),
                    constants=[1],
                    optimize='optimal')

    def cuda(self):
        """
        not sure why I need to wrap this, but I had to for the mesh information
        """
        return self.to(device='cuda')
        # super().cuda()
        # self.mesh.cuda()
        # # regenerate `oe_expr`

        # print('send to cuda')
        # if hasattr(self, 'se3'):
        #     self._oe_expr = oe.contract_expression(
        #             # subscripts;
        #             'njk,ijl,nlk -> ni',
        #             # constants;
        #             # shapes;
        #             (self.se3.shape[0],4,4),
        #             # (6,4,4),
        #             self.g,
        #             (self.se3.shape[0],4,4),
        #             constants=[1],
        #             optimize='optimal')

    @th.compile
    def forward(self):
        """
        Return:
            returns a dict with field quat, trans.
        """
        # q = torch.stack([self.qx, self.qy, self.qz, self.qw], dim=0).T
        # q = q / torch.norm(q, dim=1).reshape(-1, 1)

        # to_return = self.mesh()

        to_return = {}
        # if RTYPE == 'se3':
        # se3 = self.se3 * self.scale
        # print(SE3Exp(se3))
        # to_return['T'] = self.T0 @ SE3Exp(se3, self._oe_expr)
        



        # to_return['T'] = SE3Exp(se3) @ self.T0
        # print(to_return['T'])
        # else:
        #     # TODO add the dict from object3d to the output of the module.
        #     # to_return["quat"] = q
        #     to_return["rot6"] = self.r6
        #     # torch.stack([self.r0,
        #     #                                  self.r1,
        #     #                                  self.r2,
        #     #                                  self.r3,
        #     #                                  self.r4,
        #     #                                  self.r5], dim=-1)
        #     # to_return["trans"] = self.pos#torch.stack([self.x, self.y,
        #     # self.z], dim=0).T
        #     to_return["trans"] = torch.stack([self.x, self.y, self.z], dim=0).T
        return to_return


@dataclass
class Image:
    """
    A class to represent a image, this could be a depth image or a rgb image, etc.

    *The image has to be upside down to work in DIFF-DOPE* so the image is flipped automatically, but if you initialize it yourself you should flip it.

    Args:
        img_path (str): a path to an image to load
        img_resize (float): bounded [0,1] to resize the image
        flip_img (bool): Default is True, when initialized to the image need to be flipped (diff-dope works with flipped images)
        img_tensor (torch.tensor): an image in tensor format, assumes the image is bounded [0,1]
    """

    img_path: Optional[str] = None
    img_tensor: Optional[torch.tensor] = None
    img_resize: Optional[float] = 1
    flip_img: Optional[bool] = True
    depth: Optional[bool] = False
    depth_scale: Optional[float] = 100

    def __post_init__(self):
        if not self.img_path is None:
            if self.depth:
                im = cv2.imread(self.img_path,
                                cv2.IMREAD_UNCHANGED) / self.depth_scale
            else:
                im = cv2.imread(self.img_path)[:, :, :3]
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # normalize
                im = im / 255.0
            if self.flip_img:
                im = cv2.flip(im, 0)

            if self.img_resize < 1.0:
                if self.depth:
                    im = cv2.resize(
                        im,
                        (
                            int(im.shape[1] * self.img_resize),
                            int(im.shape[0] * self.img_resize),
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    im = cv2.resize(
                        im,
                        (
                            int(im.shape[1] * self.img_resize),
                            int(im.shape[0] * self.img_resize),
                        ),
                    )
            self.img_tensor = torch.tensor(im).float()
            log.info(
                f"Loaded image {self.img_path}, shape: {self.img_tensor.shape}")
        self._batchsize_set = False

    def __repr__(self):
        return f"{self.img_tensor.shape} @ {self.img_path} on {self.img_tensor.device}"

    def __str__(self):
        return f"{self.img_tensor.shape} @ {self.img_path} on {self.img_tensor.device}"

    def cuda(self):
        """
        Switch the img_tensor to cuda tensor
        """
        self.img_tensor = self.img_tensor.cuda().float()

    def set_batchsize(self, batchsize):
        """
        Change the batchsize for the image tensor

        Args:
            batchsize (int): batchsize for the tensor
        """
        if self._batchsize_set is False:
            self.img_tensor = torch.stack([self.img_tensor] * batchsize, dim=0)
            self._batchsize_set = True

        else:
            self.img_tensor = torch.stack(
                [self.img_tensor[0]] * batchsize, dim=0)


@dataclass
class Scene:
    """
    This class witholds images for the optimization.

    Attributes:
        tensor_rgb (torch.tensor): (batchsize,w,h,3) image tensor
        tensor_segmentation (torch.tensor): (batchsize,w,h,3) segmentation with 3 channels to facilate the computation later on.
        tensor_depth (torch.tensor): (batchsize,w,h) depth tensor (cm)
        image_resize (float): [0,1] bounding the image resize in percentage.
    """

    path_img: Optional[str] = None
    path_depth: Optional[str] = None
    path_segmentation: Optional[str] = None
    image_resize: Optional[float] = None

    tensor_rgb: Optional[Image] = None
    tensor_depth: Optional[Image] = None
    tensor_segmentation: Optional[Image] = None

    def __post_init__(self):
        # load the images and store them correctly
        if not self.path_img is None:
            self.tensor_rgb = Image(
                self.path_img, img_resize=self.image_resize)
        if not self.path_depth is None:
            self.tensor_depth = Image(
                self.path_depth, img_resize=self.image_resize, depth=True
            )
        if not self.path_segmentation is None:
            self.tensor_segmentation = Image(
                self.path_segmentation, img_resize=self.image_resize
            )

    def set_batchsize(self, batchsize):
        """
        Change the batchsize for the image tensors

        Args:
            batchsize (int): batchsize for the tensors
        """
        if not self.path_img is None:
            self.tensor_rgb.set_batchsize(batchsize)
        if not self.path_depth is None:
            self.tensor_depth.set_batchsize(batchsize)
        if not self.path_segmentation is None:
            self.tensor_segmentation.set_batchsize(batchsize)

    def get_resolution(self):
        """
        Get the scene image resolution for rendering

        Return
            (list): w,h of the image for optimization
        """
        if not self.path_img is None:
            return [
                self.tensor_rgb.img_tensor.shape[-3],
                self.tensor_rgb.img_tensor.shape[-2],
            ]
        if not self.path_depth is None:
            return [
                self.tensor_depth.img_tensor.shape[-2],
                self.tensor_depth.img_tensor.shape[-1],
            ]
        if not self.path_segmentation is None:
            return [
                self.tensor_segmentation.img_tensor.shape[-3],
                self.tensor_segmentation.img_tensor.shape[-2],
            ]

    def cuda(self):
        """
        Put on cuda the image tensors
        """

        if not self.path_img is None:
            self.tensor_rgb.cuda()
        if not self.path_depth is None:
            self.tensor_depth.cuda()
        if not self.path_segmentation is None:
            self.tensor_segmentation.cuda()


@dataclass
class DiffDope:
    """
    The main class containing all the information needed to run a Diff-DOPE optimization.
    This file is mostly driven by a config file using hydra, see the `configs/` folder.

    Args:
        cfg (DictConfig): a config file that populates the right information in the class. Please see `configs/diffdope.yaml` for more information.

    Attributes:
        optimization_results (list): a list of the different outputs from the optimization.
            Each entry is an optimization step.
            For an entry the keys are `{'rgb','depth','losses'}`
        gt_tensors (dict): a dict for `{'rgb','depth','segmentation'}' to access the torch tensor directly.
            This is useful for the image generation and for the losses defined by users.
            Moreover extent this so you can render your special losses. See examples.
        loss_functions (list): a list of function for the losses to be computed. See examples for how you could write yours.
        losses_values (dict): losses value per loss term and per image.
    """

    cfg: [DictConfig] = None
    camera: Optional[Camera] = None
    object3d: Optional[Object3D] = None
    scene: Optional[Scene] = None
    resolution: Optional[list] = None
    batchsize: Optional[int] = 16

    # TODO:
    # storing the renders for the optimization
    # how to pass add_loss
    # how to store losses
    # driven by the cfg

    def __post_init__(self):
        if self.camera is None:
            # load the camera from the config
            self.camera = Camera(**self.cfg.camera)
        # print(self.pose.position)
        if self.object3d is None:
            self.object3d = Object3D(**self.cfg.object3d)
        if self.scene is None:
            self.scene = Scene(**self.cfg.scene)
        self.batchsize = self.cfg.hyperparameters.batchsize

        # load the rendering
        self.glctx = dr.RasterizeGLContext()
        # self.glctx = dr.RasterizeCudaContext()

        self.resolution = self.scene.get_resolution()

        self.optimization_results = []

        self.gt_tensors = {}

        if self.scene.tensor_rgb is not None:
            self.gt_tensors["rgb"] = self.scene.tensor_rgb.img_tensor
        if self.scene.tensor_depth is not None:
            self.gt_tensors["depth"] = self.scene.tensor_depth.img_tensor
        if self.scene.tensor_segmentation is not None:
            self.gt_tensors["segmentation"] = self.scene.tensor_segmentation.img_tensor

        self.set_batchsize(self.batchsize)
        self.cuda()

        # Storing the values for losses display
        self.losses_values = {}

        self.loss_functions = []
        if self.cfg.losses.l1_rgb_with_mask:
            self.loss_functions.append(dd.l1_rgb_with_mask)
        if self.cfg.losses.l1_depth_with_mask:
            self.loss_functions.append(dd.l1_depth_with_mask)
        if self.cfg.losses.l1_mask:
            self.loss_functions.append(dd.l1_mask)

        # self.object3d.mesh.enable_gradients_texture()

        # logging
        log.info(f"batchsize is {self.batchsize}")
        log.info(self.object3d)
        log.info(self.scene)

    def set_batchsize(self, batchsize):
        """
        Set the batchsize for the optimization

        Args
        batchsize (int): batchsize format
        """
        self.batchsize = batchsize
        self.scene.set_batchsize(batchsize)
        self.object3d.set_batchsize(batchsize)
        self.camera.set_batchsize(batchsize)

        # todo add a cfg call here.
        # self.object3d.mesh.enable_gradients_texture()

        # self.optimizer = torch.optim.SGD(
        #     self.object3d.parameters(),
        #     lr=self.cfg.hyperparameters.learning_rate_base
        # )

        # TODO add a seed to the random
        self.learning_rates = [
            random.uniform(
                self.cfg.hyperparameters.learning_rates_bound[0],
                self.cfg.hyperparameters.learning_rates_bound[1],
            )
            for _ in range(batchsize)
        ]
        self.learning_rates = torch.tensor(self.learning_rates).float().cuda()

    def get_argmin(self):
        """
        Returns the argmin of all the losses put together
        """
        # Calculate the average tensor across all keys at the last time step
        last_time_step = -1  # Index for the last time step

        # average_tensor = torch.stack([tensor[last_time_step] for tensor in self.losses_values.values()]).mean(dim=0)

        tensor_list = []

        # Extract tensors at the last time step for each key and add to the
        # list
        for key, tensor in self.losses_values.items():
            tensor_at_last_time_step = tensor[last_time_step]
            tensor_list.append(tensor_at_last_time_step)

        # Stack the list of tensors along dimension 0 to create a single tensor
        stacked_tensor = torch.stack(tensor_list, dim=0)

        # Calculate the mean along dimension 0 to get the average tensor
        average_tensor = stacked_tensor.mean(dim=0)

        # Find the argmin of the average tensor
        argmin = torch.argmin(average_tensor, dim=-1)
        print('lr', self.learning_rates[argmin])

        return argmin

    def add_loss_value(self, key, values, values_weighted=None):
        """
        Store in `losses_values` the values of different loss term at for all the objects
            This function adds to the list of values, thus if you call multiple times on the same key, values are just added and not checked.
        Args:
            key (str): key to be store in the dict
            values (torch.tensor): B of size to be stored at key
        """

        if key not in self.losses_values:
            self.losses_values[key] = (
                values.detach().unsqueeze(0)
            )  # Create an empty tensor of the same data type
        else:
            # Append the new values (assuming values is a 1D tensor)
            self.losses_values[key] = torch.cat(
                (self.losses_values[key], values.detach().unsqueeze(0)), dim=0
            )

    def get_pose(self, batch_index=-1):
        """
        return the matrix pose as a np.ndarray

        Args:
            batch_index (int): if -1 use argmin function.

        Returns a 4x4 np.ndarray
        """
        if batch_index == -1:
            batch_index = self.get_argmin()

        matrix44 = dcn(self.optimization_results[-1]["mtx"][batch_index])

        return matrix44

    @nvtx.annotate('run_opt')
    def run_optimization(self):
        """
        If the class is set correctly this runs the optimization for finding a good pose
        """

        self.losses_values = {}
        self.optimization_results = []

        self.optimizer = torch.optim.SGD(
            self.object3d.parameters(),
            lr=self.cfg.hyperparameters.learning_rate_base
        )
        N: int = self.batchsize

        result = self.object3d.mesh()
        self.oe_expr = oe.contract_expression(
            "nij,njk,npk->npi",
            # (N, 4, 4),
            self.camera.cam_proj,
            (N, 4, 4),
            #(N, -1, 4),
            result['posw'],
            constants=[0, 2],
            optimize='optimal'
        )

        if self.scene.tensor_rgb is not None:
            self.gt_tensors["rgb"] = self.scene.tensor_rgb.img_tensor
        if self.scene.tensor_depth is not None:
            self.gt_tensors["depth"] = self.scene.tensor_depth.img_tensor
        if self.scene.tensor_segmentation is not None:
            self.gt_tensors["segmentation"] = self.scene.tensor_segmentation.img_tensor

        pbar = tqdm(range(self.cfg.hyperparameters.nb_iterations + 1))

        T0 = th.zeros((self.batchsize, 4, 4),
                      dtype=th.float32,
                      device=self.object3d.device)
        T0[..., 3, 3] = 1
        gbuf = th.ones((),
                       dtype=th.float32,
                       device=next(self.object3d.parameters()).device)
        result = self.object3d.mesh()

        for iteration_now in pbar:
            is_last_step = (iteration_now == self.cfg.hyperparameters.nb_iterations)

            with nvtx.annotate("iter"):
                itf = iteration_now / self.cfg.hyperparameters.nb_iterations + 1
                lr = (
                    self.cfg.hyperparameters.base_lr
                    * self.cfg.hyperparameters.lr_decay**itf
                )

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.zero_grad(set_to_none=False)
                #result.update(self.object3d())
                # mtx_gu = result['T']
                # se3 = self.se3 * self.scale
                # print(SE3Exp(se3))
                # to_return['T'] = self.T0 @ SE3Exp(se3, self._oe_expr)
                if True:#(not is_last_step): # fast-track
                    with torch.autograd.profiler.profile(False) as prof:
                        with nvtx.annotate("loss"):
                            (loss, bloss), mtx_gu, rgb = loss_fn(self.gt_tensors,
                                    self.camera.cam_proj,
                                    #result['T'],

                                    self.object3d.T0,
                                    self.object3d.se3,
                                    self.object3d.scale,
                                    self.object3d.g,

                                    result['posw'],
                                    result['pos_idx'],
                                    result['uv'],
                                    result['uv_idx'],
                                    result['tex'],
                                    self.learning_rates,
                                    self.cfg.losses.weight_rgb,
                                    self.resolution,
                                    self.glctx)
                            self.add_loss_value('rgb', bloss)

                    #prof.export_chrome_trace('trace.json')

                    to_add = {}
                    to_add["rgb"] = rgb.detach()  # .cpu()
                    # if self.renders['depth'] is not None:
                    #     to_add["depth"] = self.renders["depth"].detach()  # .cpu()
                    # else:
                    #     to_add["depth"] = None
                    to_add["mtx"] = mtx_gu.detach()  # .cpu()
                    self.optimization_results.append(to_add)
                else:
                    # transform quat and position into a matrix44
                    # if 'quat' in result:
                    #     mtx_gu = matrix_batch_44_from_position_quat(
                    #         p=result["trans"], q=result["quat"]
                    #     )
                    # elif 'rot6' in result:
                    #     mtx_gu = matrix_batch_44_from_position_quat(
                    #         p=result["trans"],
                    #         r6=result["rot6"]
                    #     )
                    # if RTYPE == 'se3':
                    #     mtx_gu = result['T']
                    # else:
                    #     mtx_gu = th.cat([th.cat(
                    #         [matrix_from_r6(result['rot6'] * SCALE),
                    #          result['trans'][..., :, None]], dim=-1),
                    #         T0[..., 3:, :]], dim=-2)

                    # mtx_gu = T0.slice_scatter(
                    #         matrix_from_r6(result['rot6'] * SCALE),
                    #         dim=-2, start=0, end=3)

                    mtx_gu = self.object3d.T0 @ SE3Exp(
                            self.object3d.se3 * self.object3d.scale, 
                            self.object3d.g)
                    # mtx_gu = SE3Exp(self.object3d.se3 * self.object3d.scale, 
                    #                 self.object3d.g) @ self.object3d.T0

                    with nvtx.annotate("render"):
                        if self.object3d.mesh.has_textured_map is False:
                            self.renders = render_texture_batch(
                                glctx=self.glctx,
                                proj_cam=self.camera.cam_proj,
                                mtx=mtx_gu,
                                posw=result["posw"],
                                pos_idx=result["pos_idx"],
                                vtx_color=result["vtx_color"],
                                resolution=self.resolution,
                                oe_expr=self.oe_expr
                            )
                        else:
                            # TODO test the index color version
                            self.renders = render_texture_batch(
                                glctx=self.glctx,
                                proj_cam=self.camera.cam_proj,
                                mtx=mtx_gu,
                                posw=result["posw"],
                                pos_idx=result["pos_idx"],
                                uv=result["uv"],
                                uv_idx=result["uv_idx"],
                                tex=result["tex"],
                                resolution=self.resolution,
                                oe_expr=self.oe_expr
                            )
                        to_add = {}
                        to_add["rgb"] = self.renders["rgb"].detach()  # .cpu()
                        if self.renders['depth'] is not None:
                            to_add["depth"] = self.renders["depth"].detach()  # .cpu()
                        else:
                            to_add["depth"] = None
                        to_add["mtx"] = mtx_gu.detach()  # .cpu()

                        self.optimization_results.append(to_add)

                    # computing the losses
                    with nvtx.annotate("loss"):
                        loss = torch.zeros((), device='cuda')  # .cuda()
                        for loss_function in self.loss_functions:
                            l = loss_function(self, log = is_last_step)
                            print(l.shape)
                            if l is None:
                                continue
                            loss = loss + l

                if False:
                    make_dot(loss, params=dict(self.object3d.named_parameters())).render(
                        '/tmp/docker/diffdope.gv')

                # pbar.set_description(f"loss: {loss.item():.4f}")
                gbuf.fill_(1)
                if not is_last_step:
                    with torch.autograd.profiler.profile(False) as prof:
                        with nvtx.annotate("backward"):
                            loss.backward(gradient=gbuf)
                            # loss.backward()
                    # prof.export_chrome_trace('trace.json')
                with nvtx.annotate("opt.step"):
                    self.optimizer.step()
        print(loss)

    def cuda(self):
        """
        Copy variables to the GPU.
        """
        # check the projection matrix
        #
        self.object3d.cuda()
        self.scene.cuda()
        self.camera.cuda()
        pass
