# [ICCV 2023] Deformer: Dynamic Fusion Transformer for Robust Hand Pose Estimation


The official code release for our paper
```
Deformer: Dynamic Fusion Transformer for Robust Hand Pose Estimation
Qichen Fu, Xingyu Liu, Ran Xu, Juan Carlos Niebles, Kris M. Kitani
ICCV2023
```

[[Paper](https://arxiv.org/abs/2303.04991)][[Project](https://fuqichen1998.github.io/Deformer/)][[Code](https://github.com/fuqichen1998/Deformer)]

## Setup
1. Clone this repository
    ```
    git clone https://github.com/fuqichen1998/Deformer
    ```
2. Create a python environment and install the requirements
    ```
    conda create --name deformer python=3.8
    conda activate deformer
    pip install -r requirements.txt
    ```

## Experiments

### Preparation

Download the MANO model files (`mano_v1_2.zip`) from [MANO website](http://mano.is.tue.mpg.de/). 
Unzip and put `mano/models/*` into `assets/mano_models`. 


### Training and Evaluation on HO3D Dataset

First, download and unzip [HO3D dataset](https://cloud.tugraz.at/index.php/s/9HQF57FHEQxkdcz/download?path=%2F&files=HO3D_v2.zip) to path you like, the path is referred as `$HO3D_root`. Second, download the [YCB-Objects](https://drive.google.com/file/d/1FRoMPOz0jMLimKGRdp_zGzXDiW8XnOFG) 
used in [HO3D dataset](https://www.tugraz.at/index.php?id=40231). Put unzipped folder `object_models` under `assets`. 

#### Train

We resue the processed HO3D dataset from [Semi Hand-Object](https://github.com/stevenlsw/Semi-Hand-Object/tree/master) to accelerate training. 
Please download the [preprocessed files](https://drive.google.com/file/d/1yDOJW1QbEzKjHequi-Kod1Qv6_vL_K1d). 
The downloaded files contains training list and labels generated from the [original HO3D dataset](https://cloud.tugraz.at/index.php/s/9HQF57FHEQxkdcz/download?path=%2F&files=HO3D_v2.zip). 
Please put the unzipped folder `ho3d-process` to the current directory.

Next, generate the temporal windows and the hand motion data by running:
```
python -m scripts.preprocess_ho3d
python -m scripts.generate_motions --data_root=$HO3D_root
```

Launch training by running:
```
python -m torch.distributed.launch --nproc_per_node 8 ddp_train.py \
--data_root=$HO3D_root \
--model "deformer" --train_batch=1 --test_batch=3 --workers 12 \
--T 7 --gap 10 \
--transformer_layer 3 --temporal_transformer_layer 3 \
--lr 1e-5 --motion_dis_lr 1e-3 \
--motion_discrimination \
--loss_base "maxmse" \
--snapshot 5 \
--host_folder=experiments/ho3d_train \
--run_name ho3d_train
```

The models will be automatically saved in `experiments/ho3d_train`.

#### Evaluation

Launch evaluation of a trained model by running
```
python traineval.py \
--data_root=$HO3D_root` \
--model "deformer" --test_batch=6 --workers 12 \
--T 7 --gap 10 \
--transformer_layer 3 --temporal_transformer_layer 3 \
--host_folder=experiments/ho3d_eval \
--resume TRAINED_MODEL_PATH \
--evaluate
```

The evaluation results will be saved in the `experiments/ho3d_eval`, which contains the following files: 
* `option.txt` (saved options) 
* `pred_{}.json` (```zip -j pred_{}.zip pred_{}.json``` and submit to the [official challenge](https://competitions.codalab.org/competitions/22485?) for evaluation)

### Training and Evaluation on DexYCB Dataset

First download and unzip [DexYCB dataset](https://dex-ycb.github.io/) to path you like, the path is referred as `$DexYCB_root`. 

#### Train

Run `python scripts/preprocess_dexycb.py` to preprocess the dataset. The output will be saved in the `dexycb-process`, which will be used to accelerate training.

Next, generate the hand motion data by running: 
```
python -m scripts.generate_motions --data_root=$DexYCB_root
```

Launch training by running:
```
python -m torch.distributed.launch --nproc_per_node 8 ddp_train.py \
--data_root=$DexYCB_root \
--model "deformer" --train_batch=1 --test_batch=3 --workers 12 \
--T 7 --gap 10 \
--transformer_layer 3 --temporal_transformer_layer 3 \
--lr 1e-5 --motion_dis_lr 1e-3 \
--motion_discrimination \
--loss_base "maxmse" \
--snapshot 5 \
--host_folder=experiments/dexycb_train \
--run_name dexycb_train
```

The models will be automatically saved in `experiments/dexycb_train`.

#### Evaluation

Launch evaluation of a trained model by running
```
python traineval.py \
--data_root=$DEXYCB_root` \
--model "deformer" --test_batch=6 --workers 12 \
--T 7 --gap 10 \
--transformer_layer 3 --temporal_transformer_layer 3 \
--host_folder=experiments/dexycb_eval \
--resume TRAINED_MODEL_PATH \
--evaluate
```

The evaluation results will be saved in the `experiments/dexycb_eval`, which contains the following files: 
* `option.txt` (saved options) 
* `pred_{}.json` (use the [dex-ycb-toolkit](https://github.com/NVlabs/dex-ycb-toolkit) for evaluation)

## Citation
```
@InProceedings{Fu_2023_ICCV,
    author    = {Fu, Qichen and Liu, Xingyu and Xu, Ran and Niebles, Juan Carlos and Kitani, Kris M.},
    title     = {Deformer: Dynamic Fusion Transformer for Robust Hand Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {23600-23611}
}
```


## Acknowledgments
* Semi Hand-Object https://github.com/stevenlsw/Semi-Hand-Object
* dex-ycb-toolkit https://github.com/NVlabs/dex-ycb-toolkit
* obman_train https://github.com/hassony2/obman_train
* VIBE https://github.com/mkocabas/VIBE
