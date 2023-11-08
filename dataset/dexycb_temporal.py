from dataset import dataset_util
from dataset import ho3d_util
from utils.viz_utils import write_list_to_json_file
import yaml
import torch
import os
import collections
from torchvision.transforms import functional
from torch.utils import data
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


_SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

_SERIALS = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]

_YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

_MANO_JOINTS = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]


class DexYCB_Temporal(data.Dataset):
    def __init__(
        self,
        dataset_root,
        setup="s0",
        train_label_root="./dexycb-process",
        mode="train",
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
        self._data_dir = dataset_root
        self.train_label_root = train_label_root
        self._setup = setup
        self._split = "train" if mode == "train" else "test"
        self.inp_res = inp_res
        self.T = T
        self.gap = gap
        self.coord_change_mat = np.array(
            [[1.0, 0.0, 0.0], [0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        self._model_dir = os.path.join(self._data_dir, "models")

        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self._h = 480
        self._w = 640

        if self._split == "train":
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 != 4]
        if self._split == "val":
            subject_ind = [0, 1]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        if self._split == "test":
            subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        self._subjects = [_SUBJECTS[i] for i in subject_ind]
        self._serials = [_SERIALS[i] for i in serial_ind]

        self._sequences = []
        self._mapping = []
        offset = 0
        for n in self._subjects:
            seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
            seq = [os.path.join(n, s) for s in seq]
            assert len(seq) == 100
            seq = [seq[i] for i in sequence_ind]
            self._sequences += seq
            for i, q in enumerate(seq):
                meta_file = os.path.join(self._data_dir, q, "meta.yml")
                with open(meta_file, "r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                c = np.arange(len(self._serials))
                f = np.arange(meta["num_frames"])
                f, c = np.meshgrid(f, c)
                c = c.ravel()
                f = f.ravel()
                s = (offset + i) * np.ones_like(c)
                m = np.vstack((s, c, f)).T
                self._mapping.append(m)
            offset += len(seq)
        self._mapping = np.vstack(self._mapping)

        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering
        self.max_rot = max_rot

        self.valid_idxs = []
        self.mano_params = []
        self.joints_uv = []
        self.K = []
        self.obj_p2ds = []

        self.mano_side_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_manoside.json")
        )
        mano_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_mano.json")
        )
        joints_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_joint.json")
        )
        K_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_K.json")
        )
        obj_p2d_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{self._setup}_{self._split}_obj_c2d.json")
        )
        assert len(mano_list) == len(self._mapping)
        for i in range(len(mano_list)):
            K = np.array(K_list[i], dtype=np.float32)
            self.K.append(K)
            self.joints_uv.append(
                ho3d_util.projectPoints(np.array(joints_list[i], dtype=np.float32), K)
            )
            self.mano_params.append(np.array(mano_list[i], dtype=np.float32))
            self.obj_p2ds.append(np.array(obj_p2d_list[i], dtype=np.float32))
            if (self.mano_params[-1][:48] != 0).any():
                self.valid_idxs.append(i)
        self.sequence_list = self.get_sequence_list()
        self.motions = ho3d_util.read_pkl(
            os.path.join(dataset_root, "dexycb_s0_motion_dict.pickle")
        )[self._split]

    def get_sequence_list(self):
        cache_filepath = os.path.join(
            self.train_label_root,
            f"{self._split}_temporal_windows_{self.T}_{self.gap}.json",
        )

        if not os.path.exists(cache_filepath):
            valid_idxs_set = set(self.valid_idxs)
            sequence_list = []
            for idx in self.valid_idxs:
                s, c, f = self._mapping[idx]
                # left half window
                left_window = [idx]
                prev_valid = True
                for didx in range(-1, -(self.T // 2) - 1, -1):
                    dgap = didx * self.gap
                    nidx = idx + dgap
                    # if encounter an invalid frame, or the previous frame is invalid
                    # then just repeat the previous frame
                    if nidx not in valid_idxs_set or not prev_valid:
                        left_window.append(left_window[-1])
                        prev_valid = False
                        continue
                    # if there is a sequence or camera jump
                    # then just repeat the previous frame
                    ns, nc, nf = self._mapping[nidx]
                    if ns != s or nc != c or nf != f + dgap:
                        left_window.append(left_window[-1])
                        prev_valid = False
                        continue
                    # otherwise, use the current frame
                    left_window.append(nidx)
                left_window = left_window[1:]

                # right half window
                right_window = [idx]
                prev_valid = True
                for didx in range(1, (self.T // 2) + 1, 1):
                    dgap = didx * self.gap
                    nidx = idx + dgap
                    # if encounter an invalid frame, or the previous frame is invalid
                    # then just repeat the previous frame
                    if nidx not in valid_idxs_set or not prev_valid:
                        right_window.append(right_window[-1])
                        prev_valid = False
                        continue
                    # if there is a sequence or camera jump
                    # then just repeat the previous frame
                    ns, nc, nf = self._mapping[nidx]
                    if ns != s or nc != c or nf != f + dgap:
                        right_window.append(right_window[-1])
                        prev_valid = False
                        continue
                    # otherwise, use the current frame
                    right_window.append(nidx)
                right_window = right_window[1:]

                window = left_window[::-1] + [idx] + right_window
                sequence_list.append(window)

            write_list_to_json_file(sequence_list, cache_filepath)
        return ho3d_util.json_load(cache_filepath)

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
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        # remove non-intersect object
        if (
            dataset_util.inter_area(
                dataset_util.get_bbox_joints(joints_uv),
                dataset_util.get_bbox_joints(p2d),
            )
            == 0
        ):
            crop_obj = crop_hand
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
        # Change mano from openGL coordinates to normal coordinates
        mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat)

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

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = blur_rand * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        for func in color_jitter_funcs:
            img = func(img)

        obj_mask = 0.0

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj

    def data_crop(self, img, K, bbox_hand, p2d):
        crop_hand = dataset_util.get_bbox_joints(
            bbox_hand.reshape(2, 2), bbox_factor=1.0
        )
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        if dataset_util.inter_area(bbox_hand, dataset_util.get_bbox_joints(p2d)) == 0:
            crop_obj = crop_hand
        bbox_hand = dataset_util.get_bbox_joints(
            bbox_hand.reshape(2, 2), bbox_factor=1.0
        )
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)

        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res]
        )

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)

        bbox_hand_corners = dataset_util.get_bbox_corners(bbox_hand)
        bbox_hand_corners = dataset_util.transform_coords(
            bbox_hand_corners, affinetrans
        )
        bbox_hand = dataset_util.get_bbox_joints(bbox_hand_corners)
        bbox_hand = dataset_util.regularize_bbox(bbox_hand, img.size)
        return img, K, bbox_hand, bbox_obj

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        sample = collections.defaultdict(list)
        sequence = self.sequence_list[idx]
        assert len(sequence) == self.T

        # apply the same augmentation to all frames in one sequence
        if self._split == "train":
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
            s, c, f = self._mapping[img_idx]
            d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
            img_filename = os.path.join(d, self._color_format.format(f))
            img = Image.open(img_filename).convert("RGB")

            # camera information
            K = self.K[img_idx].copy()

            # hand information
            joints_uv = self.joints_uv[img_idx].copy()
            mano_param = self.mano_params[img_idx].copy()
            mano_side = self.mano_side_list[img_idx]

            # object information
            gray = None
            p2d = self.obj_p2ds[img_idx].copy()

            if self._split == "train":
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
            else:
                bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
                img, K, bbox_hand, bbox_obj = self.data_crop(img, K, bbox_hand, p2d)
                sample["root_joint"].append(np.zeros((1, 3)))

            sample["img"].append(functional.to_tensor(img))
            sample["bbox_hand"].append(bbox_hand)
            sample["mano_param"].append(mano_param)
            sample["mano_side"].append(mano_side)
            sample["cam_intr"].append(K)
            sample["joints2d"].append(joints_uv)
            n, q = self._sequences[s].split("/")

        sample["img"] = torch.stack(sample["img"], dim=0)
        sample["bbox_hand"] = np.stack(sample["bbox_hand"], axis=0)
        sample["mano_param"] = np.stack(sample["mano_param"], axis=0)
        sample["cam_intr"] = np.stack(sample["cam_intr"], axis=0)
        sample["joints2d"] = np.stack(sample["joints2d"], axis=0)
        sample["mano_side"] = np.array(sample["mano_side"])
        if self._split != "train":
            sample["root_joint"] = np.stack(sample["root_joint"], axis=0)
        rand_real_motion_id = random.randint(0, self.motions.shape[0] - 1)
        sample["real_motion"] = self.motions[rand_real_motion_id]

        return sample
