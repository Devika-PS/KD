import mmcv.runner

from ..builder import (CLASSIFIERS, build_backbone, build_head,
                       build_neck, build_loss)
from mmcv.runner import epoch_based_runner
from ..utils.augment import Augments
from .base import BaseClassifier
import torch
import torch.nn.functional as F
from torch import nn
from shutil import ExecError
#######
from torch.autograd import Variable
from ..attacks import AttackBSS

@CLASSIFIERS.register_module()
class BaseClassifier_BSS(BaseClassifier):
    def __init__(self,
                 backbone,
                 kd_loss,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(BaseClassifier_BSS, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['student']),
                'neck': build_neck(neck['student']),
                'head': build_head(head['student'])
            }
        )

        self.teacher = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['teacher']),
                'neck': build_neck(neck['teacher']),
                'head': build_head(head['teacher'])
            }
        )

        #self.criterionMSE = torch.nn.MSELoss(size_average=False)
        self.criterionCls = F.cross_entropy
        self.criterionKD = build_loss(kd_loss)
        self.teacher_ckpt = train_cfg['teacher_checkpoint']
        self.max_epoch = train_cfg['max_epoch']
        self.load_teacher()
        self.epoch = 1
        self.count = 1
        self.attack_size = train_cfg['attack_size']
        self.ratio = 0
        self.ratio_attack = 0
        # Proposed adversarial attack algorithm (BSS)
        self.attack = AttackBSS(targeted=True, num_steps=10, max_epsilon=16, step_alpha=0.3, cuda=True, norm=2)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def load_teacher(self):
        try:
            self.teacher.load_state_dict(
                torch.load(self.teacher_ckpt)['state_dict'])
            print(f'Teacher pretrained model has been loaded {self.teacher_ckpt}')
        except:
            print(f'Teacher pretrained model has not been loaded {self.teacher_ckpt}')
        for param in self.teacher.parameters():
            param.required_grad = False

    def extract_feat(self, model, img):
        """Directly extract features from the specified stage."""

        x = model['backbone'](img)
        x = model['neck'](x)
        return x

    def get_logits(self, model, img):
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            x = x[-1]
        logit = model['head'].fc(x)  # head
        return logit

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            print("Inside augments")
            exit(0)
            img, gt_label = self.augments(img, gt_label)

        if self.epoch == 1:
            self.ratio = max(3 * (1 - self.epoch / self.max_epoch), 0) + 1
            self.ratio_attack = max(2 * (1 - 4 / 3 * self.epoch / self.max_epoch), 0) + 0

        #if(self.count == 196 & self.epoch <= self.max_epoch):
        if (self.count == 391 & self.epoch <= self.max_epoch):
            self.count = 1
            self.epoch += 1
            self.ratio = max(3 * (1 - self.epoch / self.max_epoch), 0) + 1
            self.ratio_attack = max(2 * (1 - 4 / 3 * self.epoch / self.max_epoch), 0) + 0

        batch_size1 = img.shape[0]
        inputs, targets = Variable(img), Variable(gt_label)
        #inputs, targets = img, gt_label
        t_net = self.teacher
        s_net = self.student
        out_s = self.get_logits(s_net, inputs)

        # Cross-entropy loss
        loss = self.criterionCls(out_s[0:batch_size1, :], targets)
        with torch.no_grad():
            out_t = self.get_logits(t_net, inputs)

        loss_KD = self.criterionKD(out_s, out_t.detach()) * self.ratio * (-1)/batch_size1

        if self.ratio_attack > 0:
            condition1 = targets.data == out_t.sort(dim=1, descending=True)[1][:, 0].data
            condition2 = targets.data == out_s.sort(dim=1, descending=True)[1][:, 0].data

            attack_flag = condition1 & condition2

            if attack_flag.sum():
                # Base sample selection
                attack_idx = attack_flag.nonzero().squeeze()
                if not attack_idx.shape:
                    attack_idx = torch.unsqueeze(attack_idx, 0)
                if attack_idx.shape[0] > self.attack_size:
                    diff = (F.softmax(out_t[attack_idx, :], 1).data - F.softmax(out_s[attack_idx, :], 1).data) ** 2
                    distill_score = diff.sum(dim=1) - diff.gather(1, targets[attack_idx].data.unsqueeze(1)).squeeze()
                    attack_idx = attack_idx[distill_score.sort(descending=True)[1][:self.attack_size]]

                # Target class sampling
                attack_class = out_t.sort(dim=1, descending=True)[1][:, 1][attack_idx].data
                class_score, class_idx = F.softmax(out_t, 1)[attack_idx, :].data.sort(dim=1, descending=True)
                class_score = class_score[:, 1:]
                class_idx = class_idx[:, 1:]

                rand_seed = 1 * (class_score.sum(dim=1) * torch.rand([attack_idx.shape[0]]).cuda()).unsqueeze(1)
                prob = class_score.cumsum(dim=1)
                for k in range(attack_idx.shape[0]):
                    for c in range(prob.shape[1]):
                        if (prob[k, c] >= rand_seed[k]).cpu().numpy():
                            attack_class[k] = class_idx[k, c]
                            break

                # Forward and backward for adversarial samples
                attacked_inputs = Variable(
                self.attack.run(t_net, inputs[attack_idx, :, :, :].data, attack_class))
                #attacked_inputs = self.attack.run(t_net, inputs[attack_idx, :, :, :].data, attack_class)
                batch_size2 = attacked_inputs.shape[0]

                with torch.no_grad():
                    attack_out_t = self.get_logits(t_net, attacked_inputs)
                attack_out_s = self.get_logits(s_net, attacked_inputs)

                # attack_out_t = t_net(attacked_inputs)
                # attack_out_s = s_net(attacked_inputs)

                # KD loss for Boundary Supporting Samples (BSS)
                loss_KD += self.criterionKD(attack_out_s, attack_out_t.detach()) * self.ratio_attack * (-1) / batch_size2

        #loss.backward()
        #train_loss = loss.data.item()
        train_loss = dict(loss=loss, loss_KD=loss_KD)
        #train_loss = dict(KD_loss=KD_loss, BSS_KD_loss=BSS_KD_loss)
        #train_loss = dict(loss=loss)
        #self.count += 1
        return train_loss

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(self.student, img)
        res = self.student['head'].simple_test(x)

        return res