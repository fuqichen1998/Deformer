import os
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from dataset import ho3d_util


def read_pkl(filename):
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    return data


def save_pkl(data, filename):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_dexycb(T, gap, setup="s0"):
    train_label_root = "./dexycb-process"
    motion_dict = {}
    for split in ["train", "val", "test"]:
        mano_side_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{setup}_{split}_manoside.json")
        )
        mano_list = ho3d_util.json_load(
            os.path.join(train_label_root, f"{setup}_{split}_mano.json")
        )
        mano_params = [
            np.array(mano_param, dtype=np.float32) for mano_param in mano_list
        ]
        cache_filepath = os.path.join(
            train_label_root, f"{split}_temporal_windows_{T}_{gap}.json"
        )
        sequence_list = ho3d_util.json_load(cache_filepath)
        motions = []
        for sequence in tqdm(sequence_list):
            motion = []
            for t in range(len(sequence)):
                img_idx = sequence[t]
                mano_param = mano_params[img_idx].copy()
                mano_side = mano_side_list[img_idx] * 1.0
                motion.append(np.concatenate([[mano_side], mano_param[3:48]]))
            motion = np.stack(motion, axis=0)
            motions.append(motion)
        motions = np.stack(motions, axis=0)
        motion_dict[split] = motions
    return motion_dict


def process_ho3d(T, gap):
    train_label_root = "./ho3d-process"
    sequence_list = ho3d_util.json_load(
        os.path.join(train_label_root, f"train_temporal_windows_{T}_{gap}.json")
    )
    mano_list = ho3d_util.json_load(os.path.join(train_label_root, "train_mano.json"))
    mano_params = [np.array(mano_param, dtype=np.float32) for mano_param in mano_list]
    motions = []
    for sequence in tqdm(sequence_list):
        motion = []
        for t in range(len(sequence)):
            img_idx = sequence[t]
            mano_param = mano_params[img_idx].copy()
            mano_side = 0.0
            motion.append(np.concatenate([[mano_side], mano_param[3:48]]))
        motion = np.stack(motion, axis=0)
        motions.append(motion)
    motions = np.stack(motions, axis=0)
    return {"train": motions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hand Motions")
    parser.add_argument("--data_root", type=str, help="data root path")
    args = parser.parse_args()

    T = 7
    gap = 10

    data_root = args.data_root
    if "ho3d" in data_root:
        # ho3d
        ho3d_motion_dict = process_ho3d(T=T, gap=gap)
        save_pkl(ho3d_motion_dict, os.path.join(data_root, "ho3d_motion_dict.pickle"))
    else:
        # dexycb
        dexycb_s0_motion_dict = process_dexycb(T=T, gap=gap)
        save_pkl(
            dexycb_s0_motion_dict,
            os.path.join(data_root, "dexycb_s0_motion_dict.pickle"),
        )
