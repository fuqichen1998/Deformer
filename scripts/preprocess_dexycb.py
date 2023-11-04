import os
import yaml
import json
import torch
import trimesh
import numpy as np
from tqdm.auto import tqdm

from dataset.dexycb import DexYCBDataset

dataset_root = "dex_ycb"
for setup in ["s0", "s1", "s2", "s3"]:
    for split in ["train", "val", "test"]:
        K_list = []
        joints_list = []
        mano_list = []
        mano_side_list = []  # 0 for right, 1 for left
        mano_trans_list = []
        p2d_list = []
        dataset = DexYCBDataset(dataset_root, setup, split)
        obj_mesh_dict = {k: trimesh.load(v) for k, v in dataset.obj_file.items()}
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            annotations = np.load(sample["label_file"])
            # K_list
            fx = sample["intrinsics"]["fx"]
            fy = sample["intrinsics"]["fy"]
            cx = sample["intrinsics"]["ppx"]
            cy = sample["intrinsics"]["ppy"]
            K_list.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # joints_list
            hand_joints3d = []
            for joint in annotations["joint_3d"][0]:
                hand_joints3d.append([x.item() for x in joint])
            joints_list.append(hand_joints3d)

            # mano_list
            betas = np.array(sample["mano_betas"], dtype=np.float32)
            pose_m = annotations["pose_m"][0, 0:48]
            mano_params = np.concatenate((pose_m, betas))
            mano_list.append([x.item() for x in mano_params])
            # mano_side
            if sample["mano_side"] == "right":
                mano_side_list.append(0)
            elif sample["mano_side"] == "left":
                mano_side_list.append(1)
            else:
                print(sample["mano_side"])
                raise Exception("Unknow MANO side")
            # mano_trans
            mano_trans = annotations["pose_m"][0, 48:51]
            mano_trans_list.append([x.item() for x in mano_trans])

            # object
            obj_idx = sample["ycb_grasp_ind"]
            obj_mesh = obj_mesh_dict[sample["ycb_ids"][obj_idx]]
            obj_pose = annotations["pose_y"][obj_idx]
            obj_vertices = np.array(obj_mesh.vertices)
            obj_vertices = (obj_pose[:, :3] @ obj_vertices.T + obj_pose[:, 3:]).T
            obj_vertices_proj = obj_vertices @ np.array(K_list[-1]).T
            obj_vertices_proj[:, :2] /= obj_vertices_proj[:, 2:]
            obj_vertices_proj = obj_vertices_proj[:, :2]
            min_x, min_y = obj_vertices_proj.min(0)
            max_x, max_y = obj_vertices_proj.max(0)
            obj_p2d = [
                [min_x, min_y],
                [min_x, max_y],
                [max_x, min_y],
                [max_x, max_y],
            ]
            p2d_list.append(obj_p2d)

        with open(os.path.join("dexycb-process", f"{setup}_{split}_K.json"), "w") as f:
            json.dump(K_list, f, indent=4)
        with open(
            os.path.join("dexycb-process", f"{setup}_{split}_joint.json"), "w"
        ) as f:
            json.dump(joints_list, f, indent=4)
        with open(
            os.path.join("dexycb-process", f"{setup}_{split}_mano.json"), "w"
        ) as f:
            json.dump(mano_list, f, indent=4)
        with open(
            os.path.join("dexycb-process", f"{setup}_{split}_manoside.json"), "w"
        ) as f:
            json.dump(mano_side_list, f, indent=4)
        with open(
            os.path.join("dexycb-process", f"{setup}_{split}_mano_trans.json"), "w"
        ) as f:
            json.dump(mano_trans_list, f, indent=4)
        with open(os.path.join("dexycb-process", f"{setup}_{split}_K.json"), "w") as f:
            json.dump(K_list, f, indent=4)
        with open(
            os.path.join("dexycb-process", f"{setup}_{split}_joint.json"), "w"
        ) as f:
            json.dump(joints_list, f, indent=4)
        with open(
            os.path.join("dexycb-process", f"{setup}_{split}_mano.json"), "w"
        ) as f:
            json.dump(mano_list, f, indent=4)
        with open(
            os.path.join("dexycb-process", f"{setup}_{split}_obj_c2d.json"), "w"
        ) as f:
            json.dump(p2d_list, f, indent=4)
