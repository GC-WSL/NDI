import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle as pkl
import math

class Build_NDI():
    def __init__(self, num_classes=20, length=5, tau=0.05, loss_weight=0.3):
        self.NDI_queue_length = length
        self.classes = num_classes
        self.tau = tau
        self.alpha = loss_weight
        self.NDI = [[] for i in range(self.classes)]
        self.NDI_scores = [[] for i in range(self.classes)]
    @torch.no_grad()
    def Update_NDI(self, features, obj_scores, labels_per_im):
        labels = labels_per_im[0].nonzero()
        value, idxes = torch.max(obj_scores,dim=1)
        flag = torch.ones_like(idxes).cuda()
        for label in labels:
            flag[idxes==label]=0
        flag[value<self.tau]=0
        features = features[flag==1]
        idxes = idxes[flag==1]
        obj_scores = value[flag==1]
        for f, s, idx in zip(features, obj_scores, idxes):
            if len(self.NDI[idx]) == self.NDI_queue_length:
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
    def NCL_loss(self, feats, final_score, labels_per_im):
        labels = labels_per_im[0].nonzero()
        value, idx = torch.max(final_score, dim=1)
        flag = torch.ones_like(idx).cuda()
        cosine_sim = torch.zeros_like(idx).float().cuda().detach()
        flag[value<0.001]=0
        loss = torch.tensor([0]).cuda().float()
        indexes = torch.nonzero(flag==1).cuda()    
        feats = feats[flag==1]
        idx = idx[flag==1]
        value = value[flag==1]
        count = 0
        for f, i, v in zip(feats, idx, value):
            if len(self.NDI[i])!=0:
                cos_similarity = (torch.cosine_similarity(f[None,:], torch.cat(self.NDI[i]).reshape(-1, 4096).detach())).cuda()
                sim = torch.max(cos_similarity)
                sim_id = torch.argmax(cos_similarity)
                if v >= self.tau:
                    loss += self.alpha*sim*(self.NDI_scores[i][sim_id].detach())
                else:
                    flag[indexes[count]]=0
                cosine_sim[indexes[count]] = sim.detach()
                count += 1
            else:
                flag[indexes[count]]=0
                count += 1
        return loss/(torch.sum(flag)+1), cosine_sim.detach()

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
        flag[value<self.tau]=0
        if flag.shape[0]==0 or torch.sum(flag)==0:
            loss=torch.tensor([0]).float().cuda()
        else:
            loss = -flag.detach()*(torch.log((1-value)+0.00001))
            loss =  torch.sum(loss)/torch.numel(flag.detach())
        return loss
