#!/usr/bin/env python3

from typing import Optional, Tuple, Any
from dataclasses import dataclass
import nvdiffrast.torch as dr
from enum import IntEnum
import trimesh

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops

import cv2
import kornia as K
from kornia.geometry.transform import (
    # crop_and_resize,
    crop_by_boxes
)


from lie_utils import (SE3, SE3Exp)
from camera_util import (
    ndc_projection,
    crop_and_scale_intrinsics,
    crop_params_from_mask
)
from image_loss import ImageLoss
from pkm.util.torch_util import dcn

# pos: th.Tensor
# tri: th.Tensor
# uv: Optional[th.Tensor] = None
# uvi: Optional[th.Tensor] = None
# tex: Optional[th.Tensor] = None


def _nchw(x):
    # nhwc -> nchw
    return th.moveaxis(x, -1, -3)


def _nhwc(x):
    # nchw -> nhwc
    return th.moveaxis(x, -3, -1)


def _crop_and_resize(input_tensor: th.Tensor,
                     boxes: th.Tensor,
                     size: Tuple[int, int],
                     mode: str = 'bilinear',
                     padding_mode: str = 'zeros',
                     align_corners: bool = True):
    dst_h, dst_w = size
    points_src = boxes.to(input_tensor)
    points_dst = th.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    ).expand(points_src.shape[0], -1, -1)
    return crop_by_boxes(
        input_tensor, points_src, points_dst, mode, padding_mode,
        align_corners, validate_boxes=False)


class Mesh(nn.Module):
    def __init__(self,
                 pos: th.Tensor,
                 tri: th.Tensor,
                 uv: th.Tensor,
                 uvi: th.Tensor,
                 tex: th.Tensor,
                 ):
        super().__init__()
        self.register_buffer('pos', pos, persistent=False)
        self.register_buffer('tri', tri, persistent=False)
        self.register_buffer('uv', uv, persistent=False)
        self.register_buffer('uvi', uvi, persistent=False)
        self.register_buffer('tex', tex, persistent=False)


def load_mesh(file: str, s_tex: float = 1.0,
              batch_size: Tuple[int, ...] = ()):
    """
    s_tex = texture downsampling factor
    """
    # --> pos, tri, uv, uvi, tex
    mesh = trimesh.load(file, force="mesh")

    # Convert to numpy
    pos = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.faces)
    # normals = np.asarray(mesh.vertex_normals)
    # pos_idx = torch.from_numpy(pos_idx.astype(np.int32))
    # vtx_pos = torch.from_numpy(pos.astype(np.float32)) * scale
    # vtx_normals = torch.from_numpy(normals.astype(np.float32))

    # Load texture (or vertex colors//currently unused)
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        tex = np.array(mesh.visual.material.image) / 255.0
        uv = mesh.visual.uv
        uv[:, 1] = 1 - uv[:, 1]
        # What the hell do you mean uvi == faces == tri??
        # then what's the point of this existing at all?
        uvi = np.asarray(mesh.faces)
        tex = cv2.resize(tex, dsize=None, fx=s_tex, fy=s_tex)
        tex = tex[..., :3].astype(np.float32)
    else:
        vertex_color = mesh.visual.vertex_colors[..., :3] / 255.0
        vertex_color = vertex_color.astype(np.float32)

    # == to-torch ==
    pos = th.as_tensor(pos,
                       dtype=th.float32)
    tri = th.as_tensor(tri,
                       dtype=th.int32)
    uv = th.as_tensor(uv,
                      dtype=th.float32)
    uvi = th.as_tensor(uvi,
                       dtype=th.int32)
    tex = th.as_tensor(tex,
                       dtype=th.float32)
    tex = tex.expand(*batch_size, *tex.shape).contiguous()

    # to-homogeneous
    pos = F.pad(pos, (0, 1), mode='constant', value=1)

    return Mesh(
        # Vx4
        pos=pos,
        tri=tri,
        uv=uv,
        uvi=uvi,
        tex=tex
    )


@dataclass
class RenderOutput:
    color: Optional[th.Tensor] = None
    depth: Optional[th.Tensor] = None
    label: Optional[th.Tensor] = None
    valid: Optional[th.Tensor] = None


class ColorType:  # (IntEnum):
    VERTEX = 0
    FACE = 1
    TEXTURE = 2


class ObjectRenderer(nn.Module):

    @dataclass
    class Config:
        use_color: bool = True
        use_label: bool = False
        use_depth: bool = False

        # sub-options for coloring
        # use_texture:bool = True
        # use_face_color:bool = False
        # use_vertex_color:bool = False
        # Is this even valid?
        color_type: int = ColorType.TEXTURE

    def __init__(self,
                 cfg: Config,
                 glctx=None,
                 device=None):
        super().__init__()
        self.cfg = cfg
        if glctx is None:
            glctx = dr.RasterizeGLContext(device=device)
            # self.glctx = dr.RasterizeCudaContext()
        self.glctx = glctx

    def forward(self,
                # (combined) NDC projection matrix.
                # pixel-from-camera (fixed intrinsics)
                T_pc: th.Tensor,
                # camera-from-object (batched)
                T_co: th.Tensor,
                # homogeneous vertex transforms
                # (in 'object' frame)
                posw: th.Tensor,
                # mesh face indices
                face_idx: th.Tensor,
                # __resolution__
                uv: th.Tensor,
                uv_idx: th.Tensor,
                tex: th.Tensor,

                # __vtx_color__,
                resolution: Tuple[int, int],

                # Fx|L|
                face_label: Optional[th.Tensor] = None,
                # Fx3
                face_color: Optional[th.Tensor] = None,
                # Vx3
                vertex_color: Optional[th.Tensor] = None
                ):
        cfg = self.cfg

        # i = j = 4 (homogeneous 3+1-D dimensions)
        # c : num cameras
        # ... : batch dimensions
        # v : num vertices
        # k : num points(=3)

        # But isn't pos_p obviously contiguous?
        # when going from ...cvi to (...c) v i?
        # print('proj', T_pc)
        # print('mtx', T_co)
        # raise ValueError('stop')
        # print('mtx',
        #       th.einsum('cij, ...cjk -> ...cik',
        #                 T_pc, T_co))

        pos_p = th.einsum('cij, ...cjk, vk -> ...cvi',
                          T_pc, T_co, posw)
        pos_p = pos_p.reshape(-1, *pos_p.shape[-2:])
        # .contiguous()
        # pos_p = pos_p.reshape(-1, *pos_p.shape[-2:])

        # print(pos_p.is_contiguous())
        # print(face_idx.is_contiguous())

        rast_out, rast_out_db = dr.rasterize(
            self.glctx,
            # compile-able?
            pos_p.contiguous(),
            face_idx,
            resolution=resolution
        )

        # The 'proper' depth-rendering route
        # that can propagate gradients.
        # TODO(ycho):
        # is it better to do:
        # "T_co@interpolate(posw)"
        # or "interpolate(T_co@posw)"?
        depth = None
        if cfg.use_depth:
            # NOTE(ycho):
            # I believe there's a cheaper way to
            # compute depth in case grad is not required,
            # based on rast_out[..., 2].

            # does `posw` require broadcasting too?
            # (hint: yes...probably)
            gb_pos, _ = dr.interpolate(
                posw,
                rast_out,
                face_idx,
                rast_db=rast_out_db,
                diff_attrs='all'
            )
            # TODO(ycho): optimize reshapes
            shape_keep = gb_pos.shape
            gb_pos = gb_pos.reshape(shape_keep[0], -1, shape_keep[-1])
            depth = th.matmul(gb_pos, T_co[..., 2, :, None]).squeeze(dim=-1)
            depth = -depth.reshape(shape_keep[:-1])

        label = None
        if cfg.use_label:
            if True:
                # Isn't this simpler and cheaper?
                # As long as there's gradients-- :shrug:
                # which I'm nto sure about.
                label = th.take_along_dim(face_label,
                                          rast_out[..., -1],
                                          dim=-1)
            else:
                label, _ = dr.interpolate(
                    # NOTE(ycho):
                    # By the way, does this need to be float?
                    face_label,
                    # face_label,
                    # th.ones((B,) + face_idx.shape[:-1] + (1,),
                    #        device=face_idx.device),
                    rast_out, face_idx, rast_db=rast_out_db, diff_attrs="all"
                )
            # FIXME(ycho): I don't know if this is intended,
            # but `label` is untrainable without antialias().
            label = dr.antialias(label, rast_out, pos_p, face_idx)

        color = None
        valid = None
        if cfg.use_color:
            if cfg.color_type == ColorType.VERTEX:
                color, _ = dr.interpolate(vertex_color,
                                          rast_out,
                                          face_idx,
                                          diff_attrs='all')
            elif cfg.color_type == ColorType.FACE:
                # color = th.take_along_dim(
                #        face_color, # NUM_FACE x 3
                #        rast_out[..., -1, None], # NHW
                #        dim=-2)
                color = face_color[rast_out[..., -1, None], :]
            elif cfg.color_type == ColorType.TEXTURE:
                # TODO(ycho):
                # Would it be cheaper to treat this
                # as vertex-wise attributes? Or
                # is it still better to do it like this?
                tex_c, tex_d = dr.interpolate(
                    uv, rast_out, uv_idx,
                    rast_db=rast_out_db,
                    diff_attrs='all'
                )
                color = dr.texture(
                    tex, tex_c, tex_d,
                    filter_mode='linear'
                )
            valid = (rast_out[..., -1:] > 0)

        # Do we need to return rast_out?
        # probably not, right?
        bc = (T_co.shape[:-2])
        return RenderOutput(color.reshape(bc + color.shape[1:]),
                            depth,  # .reshape(bc +depth.shape[1:]),
                            label,  # .reshape(bc +label.shape[1:]),
                            valid.reshape(bc + valid.shape[1:]))

@dataclass
class CameraConfig:
    # == render ==
    fx: Tuple[float, ...]
    fy: Tuple[float, ...]
    cx: Tuple[float, ...]
    cy: Tuple[float, ...]
    T_cw: Tuple[np.ndarray, ...]


class AlignMesh:

    @dataclass
    class Config:
        renderer: ObjectRenderer.Config = ObjectRenderer.Config()

        crop: bool = True
        # noise; (noise_rotation(radians), noise_translation(meters))
        noise: Optional[Tuple[float, float]] = None
        # relative scale of orientation params w.r.t. translation params
        # amplifies the sensitivity of rotation params during optimization.
        rscale: float = 10.0
        s_tex: float = 0.5

        # == loss ==
        loss: ImageLoss.Config = ImageLoss.Config()
        lr_bound: Tuple[float, float] = (0.1, 1.2)

        # == optimization ==
        batch_size: int = 8
        lr: float = 1.0
        base_lr: float = 20.0
        lr_decay: float = 0.1
        num_iter: int = 8

        resolution: Tuple[int, int] = (60, 80)
        z_n: float = 0.01
        z_f: float = 200.0
        mesh_file: str = ''

        device: str = 'cuda:0'

    def __init__(self, cfg: Config, cam_cfg:CameraConfig):
        self.cfg = cfg

        # Some buffers / caches
        self.g = SE3.genmat(dtype=th.float32, device=cfg.device)
        self.lr = th.zeros(cfg.batch_size,
                           dtype=th.float32,
                           device=cfg.device).uniform_(*cfg.lr_bound)
        self.se3_scale = th.as_tensor([
            cfg.rscale, cfg.rscale, cfg.rscale,
            1.0, 1.0, 1.0],
            dtype=th.float32,
            device=cfg.device)

        # 0 ~ 1
        self.grid = th.flip(th.cartesian_prod(
            th.arange(
                cfg.resolution[0],
                device=cfg.device) / cfg.resolution[0],
            th.arange(
                cfg.resolution[1],
                device=cfg.device) / cfg.resolution[1],
        ).reshape(*cfg.resolution, 2), dims=(-1,))
        # print(self.grid.min(), self.grid.max())

        # print(self.grid[0,0])
        # print(self.grid[59,79])
        # print(self.grid.shape)
        # raise ValueError('stop')

        # self.grid = F.affine_grid(
        # self.grid=th

        # Camera parameters
        # somewhat unfortunately constructed
        self.fx = th.as_tensor(cam_cfg.fx,
                               dtype=th.float32,
                               device=cfg.device)
        self.fy = th.as_tensor(cam_cfg.fy,
                               dtype=th.float32,
                               device=cfg.device)
        self.cx = th.as_tensor(cam_cfg.cx,
                               dtype=th.float32,
                               device=cfg.device)
        self.cy = th.as_tensor(cam_cfg.cy,
                               dtype=th.float32,
                               device=cfg.device)
        self.T_cw = th.as_tensor(np.stack(cam_cfg.T_cw),
                                 dtype=th.float32,
                                 device=cfg.device)

        self.renderer = ObjectRenderer(cfg.renderer, device=cfg.device)
        self.mesh = load_mesh(cfg.mesh_file,
                              s_tex=cfg.s_tex,
                              batch_size=(cfg.batch_size * len(cam_cfg.fx),),
                              ).to(cfg.device)
        self.loss = ImageLoss(cfg.loss, self.lr).to(cfg.device)

    # @th.compile
    def calc_loss(self,
                  u_se3: th.Tensor,
                  T_pc: th.Tensor,
                  T_cwo_0: th.Tensor,
                  color: th.Tensor,
                  label: th.Tensor
                  ):
        cfg = self.cfg
        T_co: th.Tensor = th.einsum('cij, ...jk -> ...cik',
                                    T_cwo_0,
                                    SE3Exp(u_se3 * self.se3_scale, self.g))

        output: RenderOutput = self.renderer(T_pc,
                                             T_co,
                                             self.mesh.pos,
                                             self.mesh.tri,
                                             self.mesh.uv,
                                             self.mesh.uvi,
                                             self.mesh.tex,
                                             cfg.resolution,
                                             # self.mesh.face_label,
                                             # self.mesh.face_color,
                                             # self.mesh.vertex_color
                                             )

        return self.loss(output.color,
                         output.depth,
                         output.label,
                         output.valid,
                         color,
                         label)

    # @th.compile
    def process_inputs(self,
                       T_wo_0: th.Tensor,
                       color: th.Tensor,
                       label: th.Tensor
                       ):
        cfg = self.cfg
        # We assume NHWC input
        # color_ud = th.flip(color, dims = (-3,))

        # It might be necessary:
        # __crop__ = find_square_crop(...)
        # or even
        # find_square_crop_that_is_multiple_of_8(...)
        # in case of using cuda ctx
        # expected resolution: (h, w) format
        crop, scale = crop_params_from_mask(label.squeeze(dim=-1),
                                            cfg.resolution,
                                            margin=0.25,
                                            match_aspect=True)
        # print(crop)
        # print(scale)

        if False:
            # Since we're going to crop the image anyway,
            # might as well flip the image here
            # _top_left, _top_right, _bottom_right, _bottom_left_
            # _xy order_
            # for now, we're going to be slightly inefficient...!
            i0, j0, di, dj = crop
            i1 = i0 + di
            j1 = j0 + dj
            bbox = th.stack([j0, i0,
                            j1, i0,
                            j1, i1,
                            j0, i1],
                            dim=-1).reshape(-1, 4, 2)

            # TODO(ycho): consider processing everything together!!
            # i.e. color = cat([color, label])
            # _CROP_AND_RESIZE_AND_FLIP_(IMGS)
            # resolution: (height, width)
            color = _nhwc(_crop_and_resize(_nchw(color), bbox,
                                           cfg.resolution))
            color = th.flip(color, dims=(-3,))

            label = _nhwc(_crop_and_resize(_nchw(label), bbox,
                                           cfg.resolution))
            label = th.flip(label, dims=(-3,))
        else:
            # grid : N H' W' 2
            i0, j0, di, dj = crop
            # S = th.stack([-di, dj], dim=-1)
            # D = th.stack([i0 + di, j0], dim=-1)
            h, w = color.shape[-3:-1]
            h *= 0.5
            w *= 0.5
            # print('h', h, 'w', w, color.shape)
            # Q = 0.5 * th.as_tensor(color.shape[-3:-1],
            #                  dtype=th.float32,
            #                  device=color.device)
            # print('Q', Q)
            S = th.stack([dj.float() / w, -di.float() / h], dim=-1)
            D = th.stack([j0 / w, (i0 + di) / h], dim=-1) - 1.0
            # print('S', 'D', S[0], D[0])
            # print('S', S.shape)
            # print('D', D.shape)
            grid = self.grid * S[:, None, None, :] + D[:, None, None, :]
            # just in case
            # grid = th.flip(grid, dims=(-1,))#[..., ::-1]
            # print('grid...')
            # print(grid[0, 0, 0], grid[0, -1, -1])
            # print(color.shape, grid.shape)
            color = _nhwc(F.grid_sample(_nchw(color), grid,
                                        mode='bilinear',  # or 'neraest' faster?
                                        padding_mode='zeros',
                                        align_corners=False))
            # print('color-minmax',
            #       color.min(),
            #       color.max(),
            #       color.shape)
            label = _nhwc(F.grid_sample(_nchw(label), grid,
                                        mode='bilinear',  # or 'neraest' faster?
                                        padding_mode='zeros',
                                        align_corners=False))

        # _CROP_AND_RESIZE_(INTRINSICS)
        fx, fy, cx, cy = crop_and_scale_intrinsics(self.fx,
                                                   self.fy,
                                                   self.cx,
                                                   self.cy,
                                                   crop,
                                                   scale)

        # FIXME(ycho): Avoid arbitrary conventions
        # like crop[3].
        T_pc = ndc_projection(cfg.resolution[1],
                              cfg.resolution[0],
                              fx,
                              fy,
                              cx,
                              cy,
                              cfg.z_n,
                              cfg.z_f)

        # camera_from_world(fixed) @
        # initial world_from_object transform
        T_wo_0 = T_wo_0.clone()
        T_wo_0[..., 1:3, :] *= -1  # for whatever reason, requires flippage
        T_cwo_0 = th.einsum('cij, jk -> cik',
                            self.T_cw,
                            T_wo_0)
        return (
            color,
            label,
            T_cwo_0,
            T_pc
        )

    def __call__(self,
                 # Initial pose transform:
                 # given as 4x4 matrix.
                 T_wo_0: th.Tensor,

                 # Color img
                 # do we really need to flip this thing?
                 color: th.Tensor,

                 # boolean segmentation mask
                 label: th.Tensor,
                 ):
        """
        color: in range (0, 1)
            (basically in same units as `texture` / `vertex_color`)
        depth: in meters, for now

        """

        cfg = self.cfg
        u_se3 = th.zeros((cfg.batch_size,
                          *T_wo_0.shape[:-2], 6),
                         dtype=T_wo_0.dtype,
                         device=T_wo_0.device,
                         requires_grad=True)

        # Optionally initialise `se3` with a bit of noise.
        if cfg.noise is not None:
            with th.no_grad():
                # u_se3[..., 0:3].normal_(0.0, cfg.noise[0] / cfg.rscale)
                # u_se3[..., 3:6].normal_(0.0, cfg.noise[1])
                pass

        # Optionally randomize learning rates
        self.lr.uniform_(*cfg.lr_bound)

        # with th.inference_mode():
        (color, label, T_cwo_0, T_pc) = self.process_inputs(T_wo_0,
                                                            color,
                                                            label)

        self.optimizer = th.optim.SGD(
            [u_se3],
            lr=cfg.lr,
        )

        # TODO(ycho):
        # Wouldn't it be more natural to
        # set batch_lr here? 0_0
        # let's test this...
        gbuf = th.ones((),
                       dtype=u_se3.dtype,
                       device=u_se3.device)

        for step in range(cfg.num_iter):
            if True:
                loss = self.calc_loss(u_se3, T_pc, T_cwo_0, color, label)
            else:
                T_co = th.einsum('cij, ...jk -> ...cik',
                                 T_cwo_0,
                                 SE3Exp(u_se3 * self.se3_scale, self.g))

                # expected resolution: (height, width)
                output: RenderOutput = self.renderer(T_pc,
                                                     T_co,
                                                     self.mesh.pos,
                                                     self.mesh.tri,
                                                     self.mesh.uv,
                                                     self.mesh.uvi,
                                                     self.mesh.tex,
                                                     cfg.resolution,
                                                     # self.mesh.face_label,
                                                     # self.mesh.face_color,
                                                     # self.mesh.vertex_color
                                                     )
                if True:
                    print(output.color.shape)  # => 8,64,64,3
                    print(output.color.min(),
                          output.color.max())
                    vis_pred = einops.rearrange(
                        dcn(output.color * output.valid),
                        'b h w c -> h (b w) c')
                    vis_true = einops.rearrange(
                        dcn(color),
                        'b h w c -> h (b w) c')
                    cv2.imshow('pred', vis_pred[::-1, :, ::-1])
                    cv2.imshow('true', vis_true[::-1, :, ::-1])
                    cv2.waitKey(0)

                loss = self.loss(output.color,
                                 output.depth,
                                 output.label,
                                 output.valid,
                                 color,
                                 label)
                print(loss)

            # [[grad step]]
            itf = step / cfg.num_iter + 1
            lr = (
                self.cfg.base_lr
                * self.cfg.lr_decay**itf
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            # == zero-grad  ==
            self.optimizer.zero_grad(set_to_none=False)

            # pbar.set_description(f"loss: {loss.item():.4f}")
            gbuf.fill_(1)
            loss.backward(gradient=gbuf)
            self.optimizer.step()
        with th.inference_mode():
            return (SE3Exp(u_se3 * self.se3_scale, self.g), loss)


def main():
    import time
    device: str = 'cuda:0'
    num_cam: int = 4

    cam_cfg = CameraConfig(
        fx=num_cam * (609.8634,),
        fy=num_cam * (609.95593,),
        cx=num_cam * (312.70883,),
        cy=num_cam * (240.19435,),
        T_cw=num_cam * (np.eye(4, dtype=np.float32),),
    )
    align = AlignMesh(AlignMesh.Config(
        mesh_file='/tmp/docker/bulldozer_normalized/simple/simple.obj',
        device=device,
    ), cam_cfg = cam_cfg)
    T_wo_0 = th.as_tensor([
        [0.6427876, 0.26198718, 0.71931569, 0.1113],
        [0., -0.939, 0.342, 0.02492],
        [0.7660444, -0.21983336, -0.60357756, 0.445977],
        [0, 0, 0, 1]
    ], dtype=th.float32, device=device)
    color = K.io.load_image('/tmp/docker/bulldozer2-imgs/002.png',
                            K.io.ImageLoadType.RGB8,
                            device=device) / 255.0
    depth = K.io.load_image('/tmp/docker/bulldozer2-imgs/depth/001.png',
                            K.io.ImageLoadType.GRAY8,
                            device=device
                            ) / 100.0
    # TODO(ycho): only temporary fix
    label = K.io.load_image('/tmp/docker/bulldozer2-imgs/mask/000.png',
                            K.io.ImageLoadType.GRAY8,
                            device=device).to(dtype=th.float32)

    # __batchify__
    color = einops.repeat(_nhwc(color), '... -> n ...', n=num_cam)
    depth = einops.repeat(_nhwc(depth), '... -> n ...', n=num_cam)
    label = einops.repeat(_nhwc(label), '... -> n ...', n=num_cam)

    t = []
    # th.cuda.synchronize(align.cfg.device)
    t.append(time.time())
    for _ in range(4):
        _, loss = align(T_wo_0, color, label)
        print(loss.item())
        # th.cuda.synchronize(align.cfg.device)
        t.append(time.time())
    # print(T_wo_0)
    # align(T_wo_0, color, label)
    # t.append(time.time())
    print(np.diff(t))


if __name__ == '__main__':
    main()
