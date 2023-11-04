import os
import collections
from torchvision.transforms import functional
from torch.utils import data
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

from dataset import ho3d_util
from dataset import dataset_util


class HO3D_Temporal(data.Dataset):
    def __init__(
        self,
        dataset_root,
        obj_model_root,
        train_label_root="./ho3d-process",
        mode="val",
        inp_res=512,
        T=5,
        gap=10,
        max_rot=np.pi,
        scale_jittering=0.2,
        center_jittering=0.1,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
    ):
        # Dataset attributes
        self.root = dataset_root
        self.mode = "train" if mode == "train" else "evaluation"
        self.inp_res = inp_res
        self.T = T
        self.gap = gap
        self.joint_root_id = 0
        self.jointsMapManoToSimple = [
            0,
            13,
            14,
            15,
            16,
            1,
            2,
            3,
            17,
            4,
            5,
            6,
            18,
            10,
            11,
            12,
            19,
            7,
            8,
            9,
            20,
        ]
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array(
            [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
        )

        # object informations
        self.obj_mesh = ho3d_util.load_objects_HO3D(obj_model_root)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)

        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot

            self.train_seg_root = os.path.join(train_label_root, "train_segLabel")

            self.mano_params = []
            self.joints_uv = []
            self.obj_p2ds = []
            self.K = []
            # training list
            self.sequence_list = ho3d_util.json_load(
                os.path.join(train_label_root, f"train_temporal_windows_{T}_{gap}.json")
            )
            self.set_list = ho3d_util.load_names(
                os.path.join(train_label_root, "train.txt")
            )
            # camera matrix
            K_list = ho3d_util.json_load(os.path.join(train_label_root, "train_K.json"))
            # hand joints
            joints_list = ho3d_util.json_load(
                os.path.join(train_label_root, "train_joint.json")
            )
            # mano params
            mano_list = ho3d_util.json_load(
                os.path.join(train_label_root, "train_mano.json")
            )
            # obj landmarks
            obj_p2d_list = ho3d_util.json_load(
                os.path.join(train_label_root, "train_obj.json")
            )
            for i in range(len(self.set_list)):
                K = np.array(K_list[i], dtype=np.float32)
                self.K.append(K)
                self.joints_uv.append(
                    ho3d_util.projectPoints(
                        np.array(joints_list[i], dtype=np.float32), K
                    )
                )
                self.mano_params.append(np.array(mano_list[i], dtype=np.float32))
                self.obj_p2ds.append(np.array(obj_p2d_list[i], dtype=np.float32))
            self.motions = ho3d_util.read_pkl(
                os.path.join(dataset_root, "ho3d_motion_dict.pickle")
            )["train"]
        else:
            self.sequence_list = ho3d_util.json_load(
                os.path.join(
                    train_label_root, f"evaluation_temporal_windows_{T}_{gap}.json"
                )
            )
            self.set_list = ho3d_util.load_names(
                os.path.join(self.root, "evaluation.txt")
            )

    def data_aug(
        self,
        img,
        mano_param,
        joints_uv,
        K,
        gray,
        p2d,
        center_rand,
        scale_rand,
        rot_rand,
        blur_rand,
        color_jitter_funcs,
    ):
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)

        # Randomly jitter center
        center_offsets = self.center_jittering * scale * center_rand
        center = center + center_offsets

        # Scale jittering
        scale_jittering = self.scale_jittering * scale_rand + 1
        scale_jittering = np.clip(
            scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering
        )
        scale = scale * scale_jittering

        rot = rot_rand
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res], rot=rot, K=K
        )
        # rotate hand root
        mano_param[:3] = dataset_util.rotation_angle(
            mano_param[:3], np.linalg.inv(rot_mat)
        )

        joints_uv = dataset_util.transform_coords(
            joints_uv, affinetrans
        )  # hand landmark trans
        K = post_rot_trans.dot(K)

        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        bbox_hand = dataset_util.regularize_bbox(bbox_hand, img.size)
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        bbox_obj = dataset_util.regularize_bbox(bbox_obj, img.size)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = blur_rand * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        for func in color_jitter_funcs:
            img = func(img)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_util.transform_img(
            gray, affinetrans, [self.inp_res, self.inp_res]
        )
        gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj

    def data_crop(self, img, K, bbox_hand, p2d):
        crop_hand = dataset_util.get_bbox_joints(
            bbox_hand.reshape(2, 2), bbox_factor=1.5
        )
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(
            bbox_hand.reshape(2, 2), bbox_factor=1.5
        )
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res]
        )
        bbox_hand = dataset_util.transform_coords(
            bbox_hand.reshape(2, 2), affinetrans
        ).flatten()
        bbox_obj = dataset_util.transform_coords(
            bbox_obj.reshape(2, 2), affinetrans
        ).flatten()
        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)
        return img, K, bbox_hand, bbox_obj

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, idx):
        sample = collections.defaultdict(list)
        sequence = self.sequence_list[idx]
        assert len(sequence) == self.T

        # apply the same augmentation to all frames in one sequence
        if self.mode == "train":
            center_rand = np.random.uniform(low=-1, high=1, size=2)
            scale_rand = np.random.randn()
            rot_rand = np.random.uniform(low=-self.max_rot, high=self.max_rot)
            blur_rand = random.random()
            color_jitter_funcs = dataset_util.get_color_jitter_funcs(
                brightness=self.brightness,
                saturation=self.saturation,
                hue=self.hue,
                contrast=self.contrast,
            )

        # preprocess each frame
        for t in range(self.T):
            img_idx = sequence[t]
            seqName, id = self.set_list[img_idx].split("/")
            img = ho3d_util.read_RGB_img(self.root, seqName, id, self.mode)
            W, H = img.size
            if self.mode == "train":
                K = self.K[img_idx].copy()
                # hand information
                joints_uv = self.joints_uv[img_idx].copy()
                mano_param = self.mano_params[img_idx].copy()
                # object information
                gray = ho3d_util.read_gray_img(self.train_seg_root, seqName, id)
                p2d = self.obj_p2ds[img_idx].copy()
                # data augmentation
                (
                    img,
                    mano_param,
                    K,
                    obj_mask,
                    p2d,
                    joints_uv,
                    bbox_hand,
                    bbox_obj,
                ) = self.data_aug(
                    img,
                    mano_param,
                    joints_uv,
                    K,
                    gray,
                    p2d,
                    center_rand,
                    scale_rand,
                    rot_rand,
                    blur_rand,
                    color_jitter_funcs,
                )
                sample["img"].append(functional.to_tensor(img))
                sample["bbox_hand"].append(bbox_hand)
                sample["mano_param"].append(mano_param)
                sample["cam_intr"].append(K)
                sample["joints2d"].append(joints_uv)
                sample["obj_p2d"].append(p2d)
                sample["obj_mask"].append(obj_mask)
            else:
                annotations = np.load(
                    os.path.join(
                        os.path.join(self.root, self.mode), seqName, "meta", id + ".pkl"
                    ),
                    allow_pickle=True,
                )
                K = np.array(annotations["camMat"], dtype=np.float32)
                # object
                sample["obj_cls"].append(annotations["objName"])
                sample["obj_bbox3d"].append(self.obj_bbox3d[annotations["objName"]])
                sample["obj_diameter"].append(
                    self.obj_diameters[annotations["objName"]]
                )
                obj_pose = ho3d_util.pose_from_RT(
                    annotations["objRot"].reshape((3,)), annotations["objTrans"]
                )
                p2d = ho3d_util.projectPoints(
                    self.obj_bbox3d[annotations["objName"]], K, rt=obj_pose
                )
                sample["obj_pose"].append(obj_pose)
                # hand
                bbox_hand = np.array(annotations["handBoundingBox"], dtype=np.float32)
                root_joint = np.array(annotations["handJoints3D"], dtype=np.float32)
                root_joint = root_joint.dot(self.coord_change_mat.T)
                sample["root_joint"].append(root_joint)
                ## clip boxes to inside image
                bbox_hand[::2] = np.clip(bbox_hand[::2], 0, W)
                bbox_hand[1::2] = np.clip(bbox_hand[1::2], 0, H)
                p2d[:, ::2] = np.clip(p2d[:, ::2], 0, W)
                p2d[:, 1::2] = np.clip(p2d[:, 1::2], 0, H)
                img, K, bbox_hand, bbox_obj = self.data_crop(img, K, bbox_hand, p2d)
                sample["img"].append(functional.to_tensor(img))
                sample["bbox_hand"].append(bbox_hand)
                sample["cam_intr"].append(K)

        # stack images and annotations in one sequence (shape: T x ...)
        if self.mode == "train":
            sample["img"] = torch.stack(sample["img"], dim=0)
            sample["bbox_hand"] = np.stack(sample["bbox_hand"], axis=0)
            sample["mano_param"] = np.stack(sample["mano_param"], axis=0)
            sample["cam_intr"] = np.stack(sample["cam_intr"], axis=0)
            sample["joints2d"] = np.stack(sample["joints2d"], axis=0)
            sample["obj_p2d"] = np.stack(sample["obj_p2d"], axis=0)
            sample["obj_mask"] = np.stack(sample["obj_mask"], axis=0)
            sample["mano_side"] = np.zeros(sample["img"].shape[0])
            rand_real_motion_id = random.randint(0, self.motions.shape[0] - 1)
            sample["real_motion"] = self.motions[rand_real_motion_id]
        else:
            sample["img"] = torch.stack(sample["img"], dim=0)
            sample["bbox_hand"] = np.stack(sample["bbox_hand"], axis=0)
            sample["cam_intr"] = np.stack(sample["cam_intr"], axis=0)
            sample["root_joint"] = np.stack(sample["root_joint"], axis=0)
            sample["obj_pose"] = np.stack(sample["obj_pose"], axis=0)
            for obj_cls in sample["obj_cls"]:
                assert (
                    obj_cls == sample["obj_cls"][0]
                ), "Object in one sequence must be the same!!!"
            sample["obj_cls"] = sample["obj_cls"][0]
            sample["obj_bbox3d"] = np.stack(sample["obj_bbox3d"], axis=0)
            sample["obj_diameter"] = np.stack(sample["obj_diameter"], axis=0)
            sample["mano_side"] = np.zeros(sample["img"].shape[0])

        return sample
