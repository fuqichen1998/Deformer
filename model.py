import torch
from torch import nn
from torchvision import ops
from networks.backbone import FPN
from networks.mano_head import HandHeatmapLayer, DynamicFusionModule
from networks.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    TransformerEncoder,
    MLP,
)
from networks.positional_encoding import (
    PositionEmbeddingSine1D,
    PositionEmbeddingSine2D,
)
from networks.loss import Joint2DLoss, ManoLoss, DynamicManoLoss


class DeformerNet(nn.Module):
    def __init__(
        self,
        roi_res=32,
        joint_nb=21,
        channels=256,
        dim_feedforward=2048,
        transformer_head=8,
        transformer_layer=6,
        temporal_transformer_head=8,
        temporal_transformer_layer=6,
        mano_layer_right=None,
        mano_layer_left=None,
        mano_neurons=[1024, 512],
        coord_change_mat=None,
        pretrained=True,
    ):
        super(DeformerNet, self).__init__()

        self.out_res = roi_res

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand transformer
        self.position_embedding = PositionEmbeddingSine2D(num_pos_feats=channels // 2)
        self.hand_query_embed = nn.Embedding(1, channels)
        self.hand_transformer = Transformer(
            d_model=channels,
            nhead=transformer_head,
            num_encoder_layers=transformer_layer,
            num_decoder_layers=transformer_layer,
            dim_feedforward=dim_feedforward,
            return_intermediate_dec=True,
        )

        # temporal hand transformer
        self.temporal_position_embedding = PositionEmbeddingSine1D(
            num_pos_feats=channels
        )
        temporal_hand_transformer_encoder_layer = TransformerEncoderLayer(
            channels, nhead=temporal_transformer_head, dim_feedforward=dim_feedforward
        )
        self.temporal_hand_transformer_encoder = TransformerEncoder(
            temporal_hand_transformer_encoder_layer,
            num_layers=temporal_transformer_layer,
        )

        self.temporal_hand_query_embed = nn.Embedding(1, channels)
        temporal_hand_transformer_decoder_layer = TransformerDecoderLayer(
            d_model=channels, nhead=transformer_head, dim_feedforward=dim_feedforward
        )
        decoder_norm = nn.LayerNorm(channels)
        self.temporal_hand_transformer_decoder = TransformerDecoder(
            temporal_hand_transformer_decoder_layer,
            num_layers=temporal_transformer_layer,
            norm=decoder_norm,
        )

        # hand head
        self.hand_hm_embed = MLP(channels, channels, joint_nb, 3)
        self.hand_hm_layer = HandHeatmapLayer(roi_res=roi_res, joint_nb=joint_nb)
        self.mano_branch = DynamicFusionModule(
            mano_layer_right,
            mano_layer_left,
            pose_feat_size=channels + 21 * 2,
            shape_feat_size=channels,
            mano_neurons=mano_neurons,
            coord_change_mat=coord_change_mat,
        )

    def net_forward(
        self,
        imgs,
        bbox_hand,
        mano_side=None,
        mano_params=None,
        roots3d=None,
        coverage=None,
    ):
        batch, T = imgs.shape[0:2]  # imgs BxTx3xHxW
        # flatten sequence from BxTx... to (B*T)x...
        imgs = imgs.flatten(0, 1)
        bbox_hand = bbox_hand.flatten(0, 1)
        if mano_side is not None:
            mano_side = mano_side.flatten(0, 1)
        if mano_params is not None:
            mano_params = mano_params.flatten(0, 1)
        if roots3d is not None:
            roots3d = roots3d.flatten(0, 1)
        if coverage is not None:
            coverage = coverage.flatten(0, 1)

        # outputs
        preds_joints = None
        pred_mano_results = []
        gt_mano_results = None

        # P2 from FPN Network
        P2 = self.base_net(imgs)[0]
        idx_tensor = torch.arange(batch * T, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)

        # here is the downscale size in FPN network(P2)
        hand_rois = ops.roi_align(
            P2,
            roi_boxes_hand,
            output_size=(self.out_res, self.out_res),
            spatial_scale=1.0 / 4.0,
            sampling_ratio=-1,
        )  # hand (B*T)xCxHxW
        # generate hand positional encoding
        hand_pos_embed = self.position_embedding(hand_rois)  # (B*T)xCxHxW

        # hand forward
        hand_hs, hand_memory = self.hand_transformer(
            hand_rois, None, self.hand_query_embed.weight, hand_pos_embed
        )  # hand_layerx(B*T)x1xC, HWx(B*T)xC
        ## 2d joints
        hand_hm = self.hand_hm_embed(hand_memory)  # HWx(B*T)x21
        hand_hm = (
            hand_hm.permute(1, 2, 0)
            .view(batch * T, 21, self.out_res, self.out_res)
            .contiguous()
        )  # (B*T)x21xHxW
        preds_joints = self.hand_hm_layer(hand_hm)  # (B*T)x21x2
        # hand_hs to mano layer
        hand_hs = hand_hs.squeeze(2)[-1]  # (B*T)xC

        # hand temporal transformer
        ## hand temporal encoder
        hand_hs = hand_hs.view(batch, T, hand_hs.shape[1]).permute(0, 2, 1)  # BxCxT
        hand_hs_pos = self.temporal_position_embedding(hand_hs)  # BxCxT
        hand_hs, hand_hs_pos = (
            hand_hs.permute(2, 0, 1).contiguous(),
            hand_hs_pos.permute(2, 0, 1).contiguous(),
        )  # TxBxC
        t_hand_memory = self.temporal_hand_transformer_encoder(
            hand_hs, pos=hand_hs_pos
        )  # TxBxC
        ## hand temporal decoder
        t_hand_query_embed = (
            self.temporal_hand_query_embed.weight.unsqueeze(1)
            .repeat(1, batch, 1)
            .contiguous()
        )  # 1xBxC
        t_hand_tgt = torch.zeros_like(t_hand_query_embed)  # 1xBxC
        t_hand_hs = self.temporal_hand_transformer_decoder(
            t_hand_tgt, t_hand_memory, pos=hand_hs_pos, query_pos=t_hand_query_embed
        )  # (1)x1xBxC
        ## final feature
        t_pose_hs = t_hand_memory.permute(1, 0, 2).flatten(0, 1)  # hand (B*T)xC
        t_pose_hs = torch.cat(
            (t_pose_hs, preds_joints.flatten(1)), dim=1
        ).contiguous()  # (B*T)x(C+42)
        t_shape_hs = t_hand_hs[0, 0]  # BxC

        ### hand temporal_hs to mano layer
        pred_mano_result, gt_mano_results = self.mano_branch(
            t_pose_hs,
            t_shape_hs,
            mano_side=mano_side,
            mano_params=mano_params,
            roots3d=roots3d,
            batch=batch,
            T=T,
            coverage=coverage,
        )
        pred_mano_results.append(pred_mano_result)

        return preds_joints, pred_mano_results, gt_mano_results

    def forward(
        self,
        imgs,
        bbox_hand,
        mano_side=None,
        mano_params=None,
        roots3d=None,
        coverage=None,
    ):
        if self.training:
            preds_joints, pred_mano_results, gt_mano_results = self.net_forward(
                imgs, bbox_hand, mano_side=mano_side, mano_params=mano_params
            )
            return preds_joints, pred_mano_results, gt_mano_results
        else:
            preds_joints, pred_mano_results, _ = self.net_forward(
                imgs, bbox_hand, mano_side=mano_side, roots3d=roots3d, coverage=coverage
            )
            return preds_joints, pred_mano_results


class Deformer(nn.Module):
    def __init__(
        self,
        honet,
        mano_lambda_verts3d=None,
        mano_lambda_joints3d=None,
        mano_lambda_manopose=None,
        mano_lambda_manoshape=None,
        mano_lambda_regulshape=None,
        mano_lambda_regulpose=None,
        lambda_joints2d=None,
        lambda_dynamic_verts3d=None,
        lambda_dynamic_joints3d=None,
        lambda_dynamic_manopose=None,
        lambda_dynamic_manoshape=None,
        lambda_end2end_verts3d=None,
        lambda_end2end_joints3d=None,
        lambda_end2end_manopose=None,
        lambda_temporal_verts3d=None,
        lambda_temporal_joints3d=None,
        lambda_temporal_manopose=None,
        lambda_temporal_manoshape=None,
        temporal_constrained=False,
        loss_base=None,
    ):
        super(Deformer, self).__init__()
        self.honet = honet
        # supervise when provide mano params
        self.mano_loss = ManoLoss(
            lambda_verts3d=mano_lambda_verts3d,
            lambda_joints3d=mano_lambda_joints3d,
            lambda_manopose=mano_lambda_manopose,
            lambda_manoshape=mano_lambda_manoshape,
            loss_base=loss_base,
        )
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        self.dynamic_loss = DynamicManoLoss(
            lambda_dynamic_verts3d=lambda_dynamic_verts3d,
            lambda_dynamic_joints3d=lambda_dynamic_joints3d,
            lambda_dynamic_manopose=lambda_dynamic_manopose,
            lambda_dynamic_manoshape=lambda_dynamic_manoshape,
            lambda_end2end_verts3d=lambda_end2end_verts3d,
            lambda_end2end_joints3d=lambda_end2end_joints3d,
            lambda_end2end_manopose=lambda_end2end_manopose,
            lambda_temporal_verts3d=lambda_temporal_verts3d,
            lambda_temporal_joints3d=lambda_temporal_joints3d,
            lambda_temporal_manopose=lambda_temporal_manopose,
            lambda_temporal_manoshape=lambda_temporal_manoshape,
            temporal_constrained=temporal_constrained,
            loss_base=loss_base,
        )
        self.temporal_constrained = temporal_constrained

    def forward(
        self,
        imgs,
        bbox_hand,
        joints_uv=None,
        joints_xyz=None,
        mano_side=None,
        mano_params=None,
        roots3d=None,
        coverage=None,
    ):
        batch, T = imgs.shape[0:2]  # imgs BxTx3xHxW
        if self.training:
            losses = {}
            total_loss = 0
            preds_joints2d, pred_mano_results, gt_mano_results = self.honet(
                imgs, bbox_hand, mano_side=mano_side, mano_params=mano_params
            )
            if mano_params is not None:
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(
                    pred_mano_results, gt_mano_results
                )
                total_loss += mano_total_loss
                for key, val in mano_losses.items():
                    losses[key] = val
            if joints_uv is not None:
                # flatten sequence from BxTx... to (B*T)x...
                joints_uv = joints_uv.flatten(0, 1).contiguous()
                joint2d_loss, joint2d_losses = self.joint2d_loss.compute_loss(
                    preds_joints2d, joints_uv
                )
                total_loss += joint2d_loss
                for key, val in joint2d_losses.items():
                    losses[key] = val
            if True:  # dynamic losses
                dynamic_loss, dynamic_losses = self.dynamic_loss.compute_loss(
                    pred_mano_results[-1], gt_mano_results, batch, T
                )
                total_loss += dynamic_loss
                for key, val in dynamic_losses.items():
                    losses[key] = val
            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0
            return total_loss, losses, pred_mano_results
        else:
            preds_joints, pred_mano_results, _ = self.honet.module.net_forward(
                imgs, bbox_hand, roots3d=roots3d, mano_side=mano_side, coverage=coverage
            )
            # only use the last layer prediction during inference
            if type(pred_mano_results) == list:
                pred_mano_results = pred_mano_results[-1]
            # already computed the averaged verts and joints in dynamic_mano_regHead
            return preds_joints, pred_mano_results
