import os
import json
from tqdm import tqdm
from dataset import ho3d_util
from utils.viz_utils import write_list_to_json_file

train_label_root = "./ho3d-process"
for mode in ["train", "evaluation"]:
    if mode == "train":
        # training list
        set_list = ho3d_util.load_names(os.path.join(train_label_root, "train.txt"))
    else:
        # evaluation list
        set_list = ho3d_util.load_names(os.path.join("ho3d_v2", "evaluation.txt"))
    # sequences
    sequences = set([x.split("/")[0] for x in set_list])
    set_set = set(set_list)
    dic = {img: i for i, img in enumerate(set_list)}

    T = 7
    gap = 10

    temporal_windows = []
    for i, img in enumerate(set_list):
        sq, fid = img.split("/")
        fid = int(fid)
        # check enough frames in temporal window
        lfid, left_window = fid, []
        for dfid in range(-1, -(T // 2) - 1, -1):
            nfid = fid + dfid * gap
            if nfid >= 0 and "/".join([sq, "%04d" % nfid]) in set_set:
                lfid = nfid
            left_window.append(lfid)

        rfid, right_window = fid, []
        for dfid in range(1, T // 2 + 1, 1):
            nfid = fid + dfid * gap
            if nfid < 9999 and "/".join([sq, "%04d" % nfid]) in set_set:
                rfid = nfid
            right_window.append(rfid)

        window = left_window[::-1] + [fid] + right_window
        temporal_window = [dic["/".join([sq, "%04d" % fidi])] for fidi in window]
        temporal_windows.append(temporal_window)

    write_list_to_json_file(
        temporal_windows,
        os.path.join(train_label_root, f"{mode}_temporal_windows_{T}_{gap}.json"),
    )
