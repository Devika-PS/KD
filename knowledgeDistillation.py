from ..builder import (CLASSIFIERS, build_backbone, build_head,
                       build_neck, build_loss)
from ..utils.augment import Augments
from .base import BaseClassifier
import torch
import torch.nn.functional as F
from torch import nn
from shutil import ExecError
import numpy as np

@CLASSIFIERS.register_module()
class KDClassifier(BaseClassifier):
    def __init__(self,
                 backbone,
                 kd_loss,
                 at_loss,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(KDClassifier, self).__init__(init_cfg)

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

        self.alpha = train_cfg['alpha']
        self.beta = train_cfg['beta']
        self.criterionKD = build_loss(kd_loss)
        self.criterionAT = build_loss(at_loss)
        self.criterionCls = F.cross_entropy
        #self.lambda_kd = train_cfg['lambda_kd']
        self.teacher_ckpt = train_cfg['teacher_checkpoint']
        self.load_teacher()

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
        #x = model['neck'](x)
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
            img, gt_label = self.augments(img, gt_label)

        with torch.no_grad():
            layer_output_teacher = self.extract_feat(self.teacher, img)
        layer_output_student = self.extract_feat(self.student, img)
        #print("only teacher", len(layer_output_teacher))
        #with torch.no_grad():

        # with torch.no_grad():
        #     teacher_logit = self.get_logits(self.teacher, img)
        # student_logit = self.get_logits(self.student, img)
        loss_AT = [self.criterionAT(x, y) for x, y in zip(layer_output_student, layer_output_teacher)]


        if isinstance(layer_output_teacher, tuple):
            layer_output_teacher = layer_output_teacher[-1]
        if isinstance(layer_output_student, tuple):
            layer_output_student = layer_output_student[-1]

        with torch.no_grad():
            layer_output_teacher = self.teacher['neck'](layer_output_teacher)
            layer_output_teacher = self.teacher['head'].fc(layer_output_teacher)

        layer_output_student = self.student['neck'](layer_output_student)
        layer_output_student = self.student['head'].fc(layer_output_student)

        loss_kl_div = self.criterionKD(layer_output_student, layer_output_teacher) * self.alpha
        loss_cls = self.criterionCls(layer_output_student, gt_label) * (1. - self.alpha)
        distillation_loss = loss_kl_div + loss_cls + self.beta * sum(loss_AT)

        losses = dict(loss_AT=sum(loss_AT),
                      distillationloss=distillation_loss)

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(self.student, img)
        x = self.student['neck'](x)
        res = self.student['head'].simple_test(x)

        return res