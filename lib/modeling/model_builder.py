from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.pcl.pcl import PCL, PCLLosses, OICR, OICRLosses, MIST, NGIS
from ops import RoIPool, RoIAlign
import modeling.pcl_heads as pcl_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.vgg_weights_helper as vgg_utils
import utils.boxes as box_utils
import pickle as pkl
import math
import numpy as np
import time
logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_MIL_Outs = pcl_heads.mil_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES)

        self.Box_Refine_Outs = pcl_heads.refine_outputs(self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.Refine_Losses = [OICRLosses() for i in range(cfg.REFINE_TIMES)]

        if cfg.MODEL.WITH_FRCNN:
            self.FRCNN_Outs = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)
        

        self._init_modules()
        #########NDI############
        self.USE_NDI = cfg.FAST_RCNN.NDI
        if self.USE_NDI:
            self.NDI_queue_lenth = 5
            self.classes = 20
            self.NDI_tau = 0.05
            self.NDI = [[] for i in range(self.classes)]
            self.NDI_scores = [[] for i in range(self.classes)]

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            if cfg.MODEL.CONV_BODY.split('.')[0] == 'vgg16':
                vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, rois, labels, step=None):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels, step)

     
    @torch.no_grad()
    def Update_NDI(self, features, obj_scores, labels_per_im):
        labels = labels_per_im[0].nonzero()
        value, idxes = torch.max(obj_scores,dim=1)
        flag = torch.ones_like(idxes).cuda()
        for label in labels:
            flag[idxes==label]=0
        flag[value<self.NDI_tau]=0
        features = features[flag==1]
        idxes = idxes[flag==1]
        obj_scores = value[flag==1]
        for f, s, idx in zip(features, obj_scores, idxes):
            if len(self.NDI[idx]) == self.NDI_queue_lenth:
                NDI_feats = self.NDI[idx]
                cos_similarity = torch.cosine_similarity(f[None,:], torch.cat(NDI_feats).reshape(-1, 4096))
                ind = torch.argmax(cos_similarity)
                feat_old = self.NDI[idx][ind]
                s_old = self.NDI_scores[idx][ind]
                ratio = (s / (s + s_old + 1e-4))
                f_new = ratio*f + (1-ratio)*feat_old
                s_new = ratio*s + (1-ratio)*s_old
                self.NDI[idx][ind] = f_new
                self.NDI_scores[idx][ind] = s_new
            else:
                self.NDI[idx].append(f)
                self.NDI_scores[idx].append(s)

                        

    def NDI_loss(self, feats, final_score, labels_per_im, alpha=0.3):
        labels = labels_per_im[0].nonzero()
        value, idx = torch.max(final_score, dim=1)
        flag = torch.ones_like(idx).cuda()
        cosine_sim = torch.zeros_like(idx).float().cuda().detach()
        flag[value<self.NDI_tau]=0
        loss = torch.tensor([0]).cuda().float()
        indexes = torch.nonzero(flag==1).cuda()    
        feats = feats[flag==1]
        idx = idx[flag==1]
        value = value[flag==1]
        count = 0
        for f, i, v in zip(feats, idx, value):
            if len(self.NDI[i])!=0:
                cos_similarity = torch.cosine_similarity(f[None,:], torch.cat(self.NDI[i]).reshape(-1, 4096).detach()).cuda()
                sim = torch.max(cos_similarity)
                sim_id = torch.argmax(cos_similarity)
                if v >= self.NDI_tau:
                    loss += alpha*sim*(self.NDI_scores[i][sim_id].detach())
                else:
                    flag[indexes[count]]=0
                cosine_sim[indexes[count]] = sim.detach()
                count += 1
            else:
                flag[indexes[count]]=0
                count += 1
        return loss/(torch.sum(flag)+1), cosine_sim

    def bce(self, pred, gt, weight=1):
        loss = -weight*(gt*(torch.log(pred+1e-6)))
        loss =  torch.sum(loss)/torch.numel(gt)
        return loss
    def NICE_loss(self, final_score, labels_per_im):
        labels = labels_per_im[0].nonzero()
        value, idx = torch.max(final_score,dim=1)
        flag = torch.ones_like(idx).cuda()
        if final_score.shape[1]==labels_per_im[0].shape[0]+1:
            flag[idx==0] = 0
            final_score = final_score[:, 1:]
            idx = idx-1
        for label in labels:
            flag[idx==label]=0
        flag[value<0.001]=0
        if flag.shape[0]==0 or torch.sum(flag)==0:
            loss=torch.tensor(0).float().cuda()
        else:
            loss = self.bce(1-value,flag.detach())
        return loss
        
   
    def _forward(self, data, rois, labels, step=None):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data).contiguous()
                   
        if not self.training:
            return_dict['blob_conv'] = blob_conv

        box_feat = self.Box_Head(blob_conv, rois)
        
        mil_score = self.Box_MIL_Outs(box_feat)
        refine_score = self.Box_Refine_Outs(box_feat)
        if cfg.MODEL.WITH_FRCNN:
            cls_score, bbox_pred = self.FRCNN_Outs(box_feat)

        device = box_feat.device

        if self.training:
            return_dict['losses'] = {}
            if self.USE_NDI:
                self.Update_NDI(box_feat, mil_score, labels)
                NCL_loss, N_Cosine =self.NDI_loss(box_feat, mil_score, labels)
                return_dict['losses']['NCL_loss'] = NCL_loss
                return_dict['losses']['loss_NICE'] = self.NICE_loss(mil_score, labels)

            # image classification loss
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            loss_im_cls = pcl_heads.mil_losses(im_cls_score, labels)
            return_dict['losses']['loss_im_cls'] = loss_im_cls

            # refinement loss
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]

            for i_refine, refine in enumerate(refine_score):
              
                if i_refine == 0:
                    if self.USE_NDI:
                        pcl_output = NGIS(boxes, mil_score, im_labels, refine, N_Cosine)
                    else:
                        pcl_output = MIST(boxes, mil_score, im_labels, refine)
                else:
                    pcl_output = MIST(boxes, refine_score[i_refine - 1], im_labels, refine)

                refine_loss = self.Refine_Losses[i_refine](
                    refine,
                    Variable(torch.from_numpy(pcl_output['labels'])).to(device),
                    Variable(torch.from_numpy(pcl_output['cls_loss_weights'])).to(device),
                    Variable(torch.from_numpy(pcl_output['gt_assignment'])).to(device))
                
                if i_refine == 0:
                    refine_loss *= 3.0

                return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()
                if self.USE_NDI:
                    return_dict['losses']['loss_NICE']+=self.NICE_loss(refine, labels)
            
            if cfg.MODEL.WITH_FRCNN:
                labels, cls_loss_weights, bbox_targets, bbox_inside_weights, \
                    bbox_outside_weights = fast_rcnn_heads.get_fast_rcnn_targets(
                        boxes, refine_score, im_labels)

                cls_loss, bbox_loss = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred,
                    Variable(torch.from_numpy(labels)).to(device),
                    Variable(torch.from_numpy(cls_loss_weights)).to(device),
                    Variable(torch.from_numpy(bbox_targets)).to(device),
                    Variable(torch.from_numpy(bbox_inside_weights)).to(device),
                    Variable(torch.from_numpy(bbox_outside_weights)).to(device))

                return_dict['losses']['cls_loss'] = cls_loss
                return_dict['losses']['bbox_loss'] = bbox_loss

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
                
        else:
            # Testing
            return_dict['rois'] = rois
            return_dict['mil_score'] = mil_score
            return_dict['refine_score'] = refine_score
            if cfg.MODEL.WITH_FRCNN:
                return_dict['cls_score'] = cls_score
                return_dict['bbox_pred'] = bbox_pred

        return return_dict
           
    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.
        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if method == 'RoIPoolF':
            xform_out = RoIPool(resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoIAlign':
            xform_out = RoIAlign(
                resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
