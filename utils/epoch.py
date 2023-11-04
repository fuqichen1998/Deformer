import os
import time
import torch
from networks.loss import batch_encoder_disc_l2_loss, batch_adv_disc_l2_loss

from utils.utils import progress_bar as bar, AverageMeters, dump


def single_epoch(
    loader,
    model,
    epoch=None,
    optimizer=None,
    save_path="checkpoints",
    motion_dis_model=None,
    motion_dis_optimizer=None,
    motion_dis_loss_weight=None,
    train=True,
    save_results=False,
    indices_order=None,
    use_cuda=False,
):
    time_meters = AverageMeters()

    if train:
        print(f"training epoch: {epoch + 1}")
        avg_meters = AverageMeters()
        model.train()

    else:
        model.eval()

        if save_results:
            # save hand results for online evaluation
            xyz_pred_list, verts_pred_list = list(), list()
            mano_shape_pred_list, mano_pose_pred_list = list(), list()
            confidences_list = list()

    end = time.time()
    for batch_idx, sample in enumerate(loader):
        if train:
            assert use_cuda and torch.cuda.is_available(), "requires cuda for training"
            imgs = sample["img"].float().cuda()
            bbox_hand = sample["bbox_hand"].float().cuda()

            mano_side = sample["mano_side"].float().cuda()
            mano_params = sample["mano_param"].float().cuda()
            joints_uv = sample["joints2d"].float().cuda()

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)
            # model forward
            model_loss, model_losses, pred_mano_results = model(
                imgs,
                bbox_hand,
                mano_side=mano_side,
                mano_params=mano_params,
                joints_uv=joints_uv,
            )

            if motion_dis_model is not None:
                if type(pred_mano_results) == list:
                    pred_mano_results = pred_mano_results[-1]

                # generator loss
                pred_motion = torch.cat(
                    [
                        mano_side[..., None],
                        pred_mano_results["expanded_mano_pose_aa"][:, :, 3:],
                    ],
                    dim=-1,
                )
                pred_motion_value = motion_dis_model(pred_motion)
                model_motion_disc_loss = (
                    batch_encoder_disc_l2_loss(pred_motion_value)
                    * motion_dis_loss_weight
                )
                model_loss += model_motion_disc_loss
                model_losses[
                    "model_motion_disc_loss"
                ] = model_motion_disc_loss.detach().cpu()
                model_losses["total_loss"] += model_motion_disc_loss.detach().cpu()

                # discriminator loss
                fake_motion = pred_motion.detach()
                real_motion = sample["real_motion"].float().cuda()
                fake_motion_value = motion_dis_model(fake_motion)
                real_motion_value = motion_dis_model(real_motion)
                (
                    motion_disc_loss_real,
                    motion_disc_loss_fake,
                    motion_disc_loss,
                ) = batch_adv_disc_l2_loss(real_motion_value, fake_motion_value)
                model_losses[
                    "motion_disc_loss_real"
                ] = motion_disc_loss_real.detach().cpu()
                model_losses[
                    "motion_disc_loss_fake"
                ] = motion_disc_loss_fake.detach().cpu()
                model_losses["motion_disc_loss"] = motion_disc_loss.detach().cpu()

                # update generator
                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()

                # update discriminator
                motion_dis_optimizer.zero_grad()
                motion_disc_loss.backward()
                motion_dis_optimizer.step()
            else:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()

            for key, val in model_losses.items():
                if val is not None:
                    avg_meters.add_loss_value(key, val)

            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            # plot progress
            suffix = (
                "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s "
                "| Mano Mesh3D Loss: {mano_mesh3d_loss:.3f} "
                "| Mano Joints3D Loss: {mano_joints3d_loss:.3f} "
                "| Mano Shape Loss: {mano_shape_loss:.3f} | Mano Pose Loss: {mano_pose_loss:.3f} "
                "| Mano Total Loss: {mano_total_loss:.3f} | Heatmap Joints2D Loss: {hm_joints2d_loss:.3f} "
                "| Temporal Mano Mesh3D Loss: {temporal_mano_mesh3d_loss:.3f} "
                "| Temporal Mano Joints3D Loss: {temporal_mano_joints3d_loss:.3f} "
                "| Temporal Mano Shape Loss: {temporal_mano_shape_loss:.3f} | Temporal Mano Pose Loss: {temporal_mano_pose_loss:.3f} "
                "| Temporal Mano Total Loss: {temporal_mano_total_loss:.3f} "
                "| Dynamic Mano Mesh3D Loss: {dynamic_mano_mesh3d_loss:.3f} "
                "| Dynamic Mano Joints3D Loss: {dynamic_mano_joints3d_loss:.3f} "
                "| Dynamic Mano Pose Loss: {dynamic_mano_pose_loss:.3f} "
                "| End2Eend Mano Mesh3D Loss: {end2end_mano_mesh3d_loss:.3f} "
                "| End2Eend Mano Joints3D Loss: {end2end_mano_joints3d_loss:.3f} "
                "| End2Eend Mano Pose Loss: {end2end_mano_pose_loss:.3f} "
                "| Dynamic Mano Total Loss: {dynamic_mano_total_loss:.3f} "
                "| Model Motion Loss: {model_motion_disc_loss:.3f} "
                "| Disc Motion Disc Real Loss: {motion_disc_loss_real:.3f} "
                "| Disc Motion Disc Fake Loss: {motion_disc_loss_fake:.3f} "
                "| Disc Motion Disc Total Loss: {motion_disc_loss:.3f} "
                "| Total Loss: {total_loss:.3f} ".format(
                    batch=batch_idx + 1,
                    size=len(loader),
                    data=time_meters.average_meters["data_time"].val,
                    bt=time_meters.average_meters["batch_time"].avg,
                    # hand losses
                    mano_mesh3d_loss=avg_meters.average_meters["mano_mesh3d_loss"].avg,
                    mano_joints3d_loss=avg_meters.average_meters[
                        "mano_joints3d_loss"
                    ].avg,
                    mano_shape_loss=avg_meters.average_meters["manoshape_loss"].avg,
                    mano_pose_loss=avg_meters.average_meters["manopose_loss"].avg,
                    mano_total_loss=avg_meters.average_meters["mano_total_loss"].avg,
                    hm_joints2d_loss=avg_meters.average_meters["hm_joints2d_loss"].avg
                    if "hm_joints2d_loss" in model_losses
                    else 0.0,
                    # temporal losses
                    temporal_mano_mesh3d_loss=avg_meters.average_meters[
                        "temporal_mano_mesh3d_loss"
                    ].avg
                    if "temporal_mano_mesh3d_loss" in model_losses
                    else 0.0,
                    temporal_mano_joints3d_loss=avg_meters.average_meters[
                        "temporal_mano_joints3d_loss"
                    ].avg
                    if "temporal_mano_joints3d_loss" in model_losses
                    else 0.0,
                    temporal_mano_shape_loss=avg_meters.average_meters[
                        "temporal_manoshape_loss"
                    ].avg
                    if "temporal_manoshape_loss" in model_losses
                    else 0.0,
                    temporal_mano_pose_loss=avg_meters.average_meters[
                        "temporal_manopose_loss"
                    ].avg
                    if "temporal_manopose_loss" in model_losses
                    else 0.0,
                    temporal_mano_total_loss=avg_meters.average_meters[
                        "temporal_mano_total_loss"
                    ].avg
                    if "temporal_mano_total_loss" in model_losses
                    else 0.0,
                    # dynamic losses
                    end2end_mano_mesh3d_loss=avg_meters.average_meters[
                        "dynamic_end2end_mesh3d_loss"
                    ].avg
                    if "dynamic_end2end_mesh3d_loss" in model_losses
                    else 0.0,
                    end2end_mano_joints3d_loss=avg_meters.average_meters[
                        "dynamic_end2end_joints3d_loss"
                    ].avg
                    if "dynamic_end2end_joints3d_loss" in model_losses
                    else 0.0,
                    end2end_mano_pose_loss=avg_meters.average_meters[
                        "dynamic_end2end_manopose_loss"
                    ].avg
                    if "dynamic_end2end_manopose_loss" in model_losses
                    else 0.0,
                    # end2end losses
                    dynamic_mano_mesh3d_loss=avg_meters.average_meters[
                        "dynamic_mano_mesh3d_loss"
                    ].avg
                    if "dynamic_mano_mesh3d_loss" in model_losses
                    else 0.0,
                    dynamic_mano_joints3d_loss=avg_meters.average_meters[
                        "dynamic_mano_joints3d_loss"
                    ].avg
                    if "dynamic_mano_joints3d_loss" in model_losses
                    else 0.0,
                    dynamic_mano_pose_loss=avg_meters.average_meters[
                        "dynamic_mano_pose_loss"
                    ].avg
                    if "dynamic_mano_pose_loss" in model_losses
                    else 0.0,
                    # dynamic total loss
                    dynamic_mano_total_loss=avg_meters.average_meters[
                        "dynamic_mano_total_loss"
                    ].avg
                    if "dynamic_mano_total_loss" in model_losses
                    else 0.0,
                    # motion_disc_loss
                    model_motion_disc_loss=avg_meters.average_meters[
                        "model_motion_disc_loss"
                    ].avg
                    if "model_motion_disc_loss" in model_losses
                    else 0.0,
                    motion_disc_loss_real=avg_meters.average_meters[
                        "motion_disc_loss_real"
                    ].avg
                    if "motion_disc_loss_real" in model_losses
                    else 0.0,
                    motion_disc_loss_fake=avg_meters.average_meters[
                        "motion_disc_loss_fake"
                    ].avg
                    if "motion_disc_loss_fake" in model_losses
                    else 0.0,
                    motion_disc_loss=avg_meters.average_meters["motion_disc_loss"].avg
                    if "motion_disc_loss" in model_losses
                    else 0.0,
                    # total loss
                    total_loss=avg_meters.average_meters["total_loss"].avg,
                )
            )
            bar(suffix)
            end = time.time()

        else:
            assert use_cuda and torch.cuda.is_available(), "requires cuda for testing"
            if use_cuda and torch.cuda.is_available():
                imgs = sample["img"].float().cuda()
                bbox_hand = sample["bbox_hand"].float().cuda()
                mano_side = sample["mano_side"].float().cuda()
                if "coverage" in sample:
                    coverage = sample["coverage"].float().cuda()
                else:
                    coverage = None
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float().cuda()
                else:
                    root_joints = None

            else:
                imgs = sample["img"].float()
                bbox_hand = sample["bbox_hand"].float()
                mano_side = sample["mano_side"].float()
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float()
                else:
                    root_joints = None

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)
            preds_joints, results = model(
                imgs, bbox_hand, mano_side=mano_side, roots3d=root_joints
            )
            pred_xyz = results["joints3d"].detach().cpu().numpy()
            pred_verts = results["verts3d"].detach().cpu().numpy()
            pred_mano_shape = results["mano_shape"].detach().cpu().numpy()
            pred_mano_pose = results["mano_pose_aa"].detach().cpu().numpy()

            if save_results:
                for xyz, verts, mano_shape, mano_pose in zip(
                    pred_xyz, pred_verts, pred_mano_shape, pred_mano_pose
                ):
                    if indices_order is not None:
                        xyz = xyz[indices_order]
                    xyz_pred_list.append(xyz)
                    verts_pred_list.append(verts)
                    mano_shape_pred_list.append(mano_shape)
                    mano_pose_pred_list.append(mano_pose)
                if "confidences" in results:
                    confidences = results["confidences"].detach().cpu().numpy()
                    for confidence in confidences:
                        confidences_list.append(confidence)

            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s".format(
                batch=batch_idx + 1,
                size=len(loader),
                data=time_meters.average_meters["data_time"].val,
                bt=time_meters.average_meters["batch_time"].avg,
            )

            bar(suffix)
            end = time.time()

    if train:
        return avg_meters
    else:
        if save_results:
            pred_out_path = os.path.join(
                save_path,
                "pred_epoch_{}.json".format(epoch + 1)
                if epoch is not None
                else "pred_{}.json",
            )
            dump(pred_out_path, xyz_pred_list, verts_pred_list)
            pred_out_path = os.path.join(
                save_path,
                "pred_mano_epoch_{}.json".format(epoch + 1)
                if epoch is not None
                else "pred_mano_{}.json",
            )
            dump(pred_out_path, mano_shape_pred_list, mano_pose_pred_list)
            if confidences_list:
                pred_out_path = os.path.join(
                    save_path,
                    "pred_confidences_epoch_{}.json".format(epoch + 1)
                    if epoch is not None
                    else "pred_confidences_{}.json",
                )
                dump(pred_out_path, confidences_list, confidences_list)
        return None
