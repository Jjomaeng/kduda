#---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform,target_strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmcv.runner import  load_checkpoint
from mmseg.models.utils.proto_estimator import ProtoEstimator
from mmseg.models.losses.contrastive_loss import contrast_preparations,bank_contrastive
import time


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}

        # feature storage for contrastive
        self.feat_distributions = None
        self.ignore_index = 255
        self.start_distribution_iter = 50

        #mit-b3 student model generate
        #std_cfg = deepcopy(cfg['model'])
        #std_cfg['pretrained'] = 'pretrained/mit_b3.pth'
        #self.std_model = build_student(std_cfg)

        #mit-b5 teacher model (pretrained) generate
        self.teacher_cfg = deepcopy(cfg['model'])
        self.teacher_cfg ['pretrained'] = None
        self.teacher_cfg ['backbone'] = {'type': 'mit_b5', 'style': 'pytorch', 'drop_path_rate': 0.1}
        self.teacher_cfg ['decode_head']['decoder_params'] = {'embed_dims': 256, 'output_stride': 4, 'embed_cfg': {'type': 'mlp', 'act_cfg': None, 'norm_cfg': None}, 'embed_neck_cfg': {'type': 'mlp', 'act_cfg': None, 'norm_cfg': None}, 'fusion_operation': 'cat', 'fusion_cfg': {'type': 'aspp', 'sep': True, 'dilations': [1, 6, 12, 18], 'pool': False, 'act_cfg': {'type': 'ReLU'}, 'norm_cfg': {'type': 'BN', 'requires_grad': True}}, 'head_cfg': None, 'head_num': 0}
        self.teacher_cfg ['train_cfg'] = None
        self.teacher_cfg ['auxiliary_head'] = None



        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.feat_distributions = ProtoEstimator(dim=512, class_num=19,memory_length=200)

    def get_teacher_model(self):
        return get_module(self.tea_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)



    def train_step(self, data_batch, optimizer, **kwargs):

        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):


        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components

        """
        self.tea_model = build_segmentor(self.teacher_cfg, test_cfg=None)
        checkpoint_pth = 'work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth'
        checkpoint = load_checkpoint(
            self.tea_model,
            checkpoint_pth,
            map_location='cpu',
            revise_keys=[(r'^module\.', ''), ('model.', '')])

        self.tea_model.to("cuda")


        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model


        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_teacher_model().modules():

            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_teacher_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight_= torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight_ * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)
        target_masks = get_class_masks(pseudo_label)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        #teacehr - student KL loss

        #make source + target original image
        # ori_img = [None] * batch_size
        # for i in range(batch_size):
        #     strong_parameters['mix'] = mix_masks[i]
        #     _, ori_img[i] = strong_transform(
        #         strong_parameters,
        #         target=torch.stack((img[i], target_img[i]))) #label update
        #
        # ori_img = torch.cat(ori_img)

            #augment target image
        strong_parameters['mix_target'] = target_masks
        aug_target_img,ori_target_img = target_strong_transform(strong_parameters,target_img)

        #bank stroage
        bank = {}

    #    augment_kl_loss = self.get_model().forward_train( target_img, target_img_metas, gt_semantic_seg, return_feat=True) # feature space
    #    src_kl_feat = augment_kl_loss.pop('features') #encoder_decoder.py -> extract_feat(target_img)
        target_kl_feat = self.get_model().encode_decode( aug_target_img, target_img_metas)
    #    student_src_feat = self.get_model().extract_feat(img)##output space
        student_trg_feat = self.get_model().extract_decode_context(target_img,target_img_metas,return_context = True)# for cl loss


        with torch.no_grad(): #teacher
            tea_target_feat = self.get_teacher_model().encode_decode(ori_target_img, target_img_metas)
            tea_trg_feat = self.get_teacher_model().extract_decode_context(target_img,target_img_metas,return_context = True)
        #    tea_trg_feat = self.get_model().auxiliary_project_feat(x) #for cl loss
        #     tea_src_feat = self.get_teacher_model().encode_decode(img, img_metas) #output space
        #     tea_feat = self.get_teacher_model().extract_feat(target_img) #feature space

        #bank update : original target image - teacher network output
        pseudo_label_cl = pseudo_label.view(2,1,512,512) # variable change !!
        feat, mask = contrast_preparations(tea_trg_feat,pseudo_label_cl,True,0.75,19,255)
        self.feat_distributions.update_proto(features=feat.detach(), labels=mask)
        bank = self.feat_distributions.MemoryBank

        #contrastive loss
        if self.local_iter >= self.start_distribution_iter:
            cl_loss = bank_contrastive(student_trg_feat,pseudo_label_cl,bank)
            cl_loss, _ = self._parse_losses({'contrastive loss': cl_loss})
            cl_loss.backward()

        #target kl loss
        B,C,h,w = tea_target_feat.size()
        scale_pred_trg = target_kl_feat.permute(0,2,3,1).contiguous().view(-1,C) #student
        scale_soft_trg = tea_target_feat.permute(0,2,3,1).contiguous().view(-1,C) #teacher
        p_s_trg = F.log_softmax(scale_pred_trg,dim = 1)
        p_t_trg = F.softmax(scale_soft_trg,dim = 1)
        kl_loss_trg = F.kl_div(p_s_trg,p_t_trg,reduction='batchmean')
        kl_loss_trg = pseudo_weight_ * kl_loss_trg
        kl_loss_trg,_ = self._parse_losses({'KL_div_loss_trg': kl_loss_trg})
        kl_loss_trg.backward()

        # # source kl loss
        # scale_pred_src = src_kl_feat.permute(0, 2, 3, 1).contiguous().view(-1, C)  # student
        # scale_soft_src = tea_src_feat.permute(0, 2, 3, 1).contiguous().view(-1, C)  # teacher
        # p_s = F.log_softmax(scale_pred_src, dim=1)
        # p_t = F.softmax(scale_soft_src, dim=1)
        # kl_loss_src = F.kl_div(p_s, p_t, reduction='batchmean')
        # kl_loss_src, _ = self._parse_losses({'KL_div_loss_src': kl_loss_src})
        # kl_loss_src.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(aug_target_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed target Image')
                subplotimg(
                    axs[1][2], target_masks[j][0], 'target Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars