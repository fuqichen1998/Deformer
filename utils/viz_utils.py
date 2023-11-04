import os
import json
import torch
import imageio
import trimesh
import pyrender

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_list_to_json_file(list, filename):
    with open(filename, "w") as f:
        list_str = json.dumps(list)
        f.write(list_str)


def show_img(img):
    plt.imshow(img)
    plt.show()


def save_img(img, save_path):
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_scene(
    hand_vert,
    cam_intr,
    mano_layer,
    coord_change_mat=torch.eye(3),
    hand_verts_color=None,
    wire_frame=False,
    black_bg=False,
):
    # Create pyrender scene.
    if black_bg:
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
            ambient_light=np.array([1.0, 1.0, 1.0]),
        )
    else:
        scene = pyrender.Scene(
            bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
            ambient_light=np.array([1.0, 1.0, 1.0]),
        )

    # add camera
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(cam, pose=np.eye(4))

    # add hand mesh
    faces = mano_layer.th_faces.numpy()
    # predictions
    vert = torch.tensor(hand_vert) @ coord_change_mat
    mesh = trimesh.Trimesh(vertices=vert, faces=faces, vertex_colors=hand_verts_color)
    mesh1 = pyrender.Mesh.from_trimesh(mesh)
    if wire_frame:
        mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
        mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
        mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
        node1 = scene.add(mesh1)
        node2 = scene.add(mesh2)
    else:
        node1 = scene.add(mesh1)
    return scene
