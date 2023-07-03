# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
from collections import deque


class ProtoEstimator:
    def __init__(self, dim, class_num, memory_length=100, resume=""):
        super(ProtoEstimator, self).__init__()
        self.dim = dim
        self.class_num = class_num

        # init mean and covariance
        if resume:
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.CoVariance = checkpoint['CoVariance'].cuda()
            self.Ave = checkpoint['Ave'].cuda()
            self.Amount = checkpoint['Amount'].cuda()
            if 'MemoryBank' in checkpoint:
                self.MemoryBank = checkpoint['MemoryBank'].cuda()
        else:
            self.CoVariance = torch.zeros(self.class_num, self.dim).cuda()
            self.Ave = torch.zeros(self.class_num, self.dim).cuda()
            self.Amount = torch.zeros(self.class_num).cuda()
            self.MemoryBank = [deque([self.Ave[cls].unsqueeze(0).detach()], maxlen=memory_length)
                               for cls in range(self.class_num)]

    def update_proto(self, features, labels):
        """Update variance and mean

        Args:
            features (Tensor): feature map, shape [B, A, H, W]  N = B*H*W
            labels (Tensor): shape [B, 1, H, W]
        """

        N, A = features.size()
        C = self.class_num

        NxCxA_Features = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxA_Features.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        # update memory bank
        for cls in torch.unique(labels):
            self.MemoryBank[cls].append(ave_CxA[cls].unsqueeze(0).detach())

