import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.losses import *
from models.encoder import Encoder
from models.modeling.deeplab import DeepLab as DeepLab_v3p
import numpy as np


class CAC(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(CAC, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))

        self.ignore_index = ignore_index

        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']
        self.proj_final_dim = conf['proj_final_dim']

        assert self.layers in [50, 101]

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

        if self.mode == 'semi':
            self.project = nn.Sequential(
                nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_dim, self.proj_final_dim, kernel_size=1, stride=1)
            )

            self.weight_unsup = conf['weight_unsup']
            self.temp = conf['temp']
            self.epoch_start_unsup = conf['epoch_start_unsup']
            self.selected_num = conf['selected_num']
            self.step_save = conf['step_save']
            self.step_count = 0
            self.feature_bank = []
            self.pseudo_label_bank = []
            self.pos_thresh_value = conf['pos_thresh_value']
            self.stride = conf['stride']

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, gpu=None, gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None):
        if not self.training:
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

        if self.mode == 'supervised':
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index, temperature=1.0) * self.sup_loss_w
            
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

        elif self.mode == 'semi':
            
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index, temperature=1.0) * self.sup_loss_w
            
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            
            if epoch < self.epoch_start_unsup:
                return total_loss, curr_losses, outputs

            # x_ul: [batch_size, 2, 3, H, W]
            x_ul1 = x_ul[:, 0, :, :, :]
            x_ul2 = x_ul[:, 1, :, :, :]

            enc_ul1 = self.encoder(x_ul1)
            if self.downsample:
                enc_ul1 = F.avg_pool2d(enc_ul1, kernel_size=2, stride=2)
            output_ul1 = self.project(enc_ul1) #[b, c, h, w]
            output_ul1 = F.normalize(output_ul1, 2, 1)

            enc_ul2 = self.encoder(x_ul2)
            if self.downsample:
                enc_ul2 = F.avg_pool2d(enc_ul2, kernel_size=2, stride=2)
            output_ul2 = self.project(enc_ul2) #[b, c, h, w]
            output_ul2 = F.normalize(output_ul2, 2, 1)

            # compute pseudo label
            logits1 = self.classifier(enc_ul1) #[batch_size, num_classes, h, w]
            logits2 = self.classifier(enc_ul2)
            pseudo_logits_1 = F.softmax(logits1, 1).max(1)[0].detach() #[batch_size, h, w]
            pseudo_logits_2 = F.softmax(logits2, 1).max(1)[0].detach()          
            pseudo_label1 = logits1.max(1)[1].detach() #[batch_size, h, w]
            pseudo_label2 = logits2.max(1)[1].detach()

            # get overlap part
            output_feature_list1 = []
            output_feature_list2 = []
            pseudo_label_list1 = []
            pseudo_label_list2 = []
            pseudo_logits_list1 = []
            pseudo_logits_list2 = []
            for idx in range(x_ul1.size(0)):
                output_ul1_idx = output_ul1[idx]
                output_ul2_idx = output_ul2[idx]
                pseudo_label1_idx = pseudo_label1[idx]
                pseudo_label2_idx = pseudo_label2[idx]
                pseudo_logits_1_idx = pseudo_logits_1[idx]
                pseudo_logits_2_idx = pseudo_logits_2[idx]
                if flip[0][idx] == True:
                    output_ul1_idx = torch.flip(output_ul1_idx, dims=(2,))
                    pseudo_label1_idx = torch.flip(pseudo_label1_idx, dims=(1,))
                    pseudo_logits_1_idx = torch.flip(pseudo_logits_1_idx, dims=(1,))
                if flip[1][idx] == True:
                    output_ul2_idx = torch.flip(output_ul2_idx, dims=(2,))
                    pseudo_label2_idx = torch.flip(pseudo_label2_idx, dims=(1,))
                    pseudo_logits_2_idx = torch.flip(pseudo_logits_2_idx, dims=(1,))
                output_feature_list1.append(output_ul1_idx[:, ul1[0][idx]//8:br1[0][idx]//8, ul1[1][idx]//8:br1[1][idx]//8].permute(1, 2, 0).contiguous().view(-1, output_ul1.size(1)))
                output_feature_list2.append(output_ul2_idx[:, ul2[0][idx]//8:br2[0][idx]//8, ul2[1][idx]//8:br2[1][idx]//8].permute(1, 2, 0).contiguous().view(-1, output_ul2.size(1)))
                pseudo_label_list1.append(pseudo_label1_idx[ul1[0][idx]//8:br1[0][idx]//8, ul1[1][idx]//8:br1[1][idx]//8].contiguous().view(-1))
                pseudo_label_list2.append(pseudo_label2_idx[ul2[0][idx]//8:br2[0][idx]//8, ul2[1][idx]//8:br2[1][idx]//8].contiguous().view(-1))
                pseudo_logits_list1.append(pseudo_logits_1_idx[ul1[0][idx]//8:br1[0][idx]//8, ul1[1][idx]//8:br1[1][idx]//8].contiguous().view(-1))
                pseudo_logits_list2.append(pseudo_logits_2_idx[ul2[0][idx]//8:br2[0][idx]//8, ul2[1][idx]//8:br2[1][idx]//8].contiguous().view(-1))
            output_feat1 = torch.cat(output_feature_list1, 0) #[n, c]
            output_feat2 = torch.cat(output_feature_list2, 0) #[n, c]
            pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0) #[n,]
            pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0) #[n,]
            pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0) #[n,]
            pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0) #[n,]  
            assert output_feat1.size(0) == output_feat2.size(0)
            assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
            assert output_feat1.size(0) == pseudo_label1_overlap.size(0)

            # concat across multi-gpus
            b, c, h, w = output_ul1.size()
            selected_num = self.selected_num
            output_ul1_flatten = output_ul1.permute(0, 2, 3, 1).contiguous().view(b*h*w, c)
            output_ul2_flatten = output_ul2.permute(0, 2, 3, 1).contiguous().view(b*h*w, c)
            selected_idx1 = np.random.choice(range(b*h*w), selected_num, replace=False)
            selected_idx2 = np.random.choice(range(b*h*w), selected_num, replace=False)
            output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
            output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
            output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected], 0) #[2*kk, c]
            output_ul_all = self.concat_all_gather(output_ul_flatten_selected) #[2*N, c]

            pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
            pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
            pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected], 0) #[2*kk]
            pseudo_label_all = self.concat_all_gather(pseudo_label_flatten_selected) #[2*N]

            self.feature_bank.append(output_ul_all)
            self.pseudo_label_bank.append(pseudo_label_all)
            if self.step_count > self.step_save:
                self.feature_bank = self.feature_bank[1:]
                self.pseudo_label_bank = self.pseudo_label_bank[1:]
            else:
                self.step_count += 1
            output_ul_all = torch.cat(self.feature_bank, 0)
            pseudo_label_all = torch.cat(self.pseudo_label_bank, 0)

            eps = 1e-8
            pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp #[n, 1]
            pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp #[n, 1]

            # compute loss1
            b = 8000
            def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
                # print("gpu: {}, i_1: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
                return logits1_neg_idx

            def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
                # print("gpu: {}, i_1_0: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
                neg1_idx = torch.cat([pos, neg1_idx], 1) #[n, 1+b]
                mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).float().cuda(), mask1_idx], 1) #[n, 1+b]
                neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0] #[n, 1]
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
                return logits1_neg_idx, neg_max1

            N = output_ul_all.size(0)
            logits1_down = torch.zeros(pos1.size(0)).float().cuda()
            for i in range((N-1)//b + 1):
                # print("gpu: {}, i: {}".format(gpu, i))
                pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
                output_ul_idx = output_ul_all[i*b:(i+1)*b]
                if i == 0:
                    logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap)
                else:
                    logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1)
                logits1_down += logits1_neg_idx

            logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)

            pos_mask_1 = ((pseudo_logits2_overlap > self.pos_thresh_value) & (pseudo_logits1_overlap < pseudo_logits2_overlap)).float()
            loss1 = -torch.log(logits1 + eps)
            loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)

            # compute loss2
            def run2(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2):
                # print("gpu: {}, i_2: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
                return logits2_neg_idx

            def run2_0(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap):
                # print("gpu: {}, i_2_0: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
                neg2_idx = torch.cat([pos, neg2_idx], 1) #[n, 1+b]
                mask2_idx = torch.cat([torch.ones(mask2_idx.size(0), 1).float().cuda(), mask2_idx], 1) #[n, 1+b]
                neg_max2 = torch.max(neg2_idx, 1, keepdim=True)[0] #[n, 1]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
                return logits2_neg_idx, neg_max2

            N = output_ul_all.size(0)
            logits2_down = torch.zeros(pos2.size(0)).float().cuda()
            for i in range((N-1)//b + 1):
                pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
                output_ul_idx = output_ul_all[i*b:(i+1)*b]
                if i == 0:
                    logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(run2_0, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap)
                else:
                    logits2_neg_idx = torch.utils.checkpoint.checkpoint(run2, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2)
                logits2_down += logits2_neg_idx

            logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)

            pos_mask_2 = ((pseudo_logits1_overlap > self.pos_thresh_value) & (pseudo_logits2_overlap < pseudo_logits1_overlap)).float()

            loss2 = -torch.log(logits2 + eps)
            loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)

            loss_unsup = self.weight_unsup * (loss1 + loss2)
            curr_losses['loss1'] = loss1
            curr_losses['loss2'] = loss2
            curr_losses['loss_unsup'] = loss_unsup
            total_loss = total_loss + loss_unsup
            return total_loss, curr_losses, outputs

        else:
            raise ValueError("No such mode {}".format(self.mode))

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'supervised':
            return chain(self.encoder.get_module_params(), self.classifier.parameters())
        elif self.mode == 'semi':
            return chain(self.encoder.get_module_params(), self.classifier.parameters(), self.project.parameters())
        else:
            raise ValueError("No such mode {}".format(self.mode))
