"""
    Author: He Jiaxin
    Date: 2019/07/06
    Version: 1.0
    Function: define 3 kind of elements in autodeeplab: Node, Cell, Network
"""

from operators import *
from genotypes import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Node(nn.Module):
    def __init__(self, C, stride):
        super(Node, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False, track_running_stats=False))
            self._ops.append(op)
    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

DIFF_STATE_OPS = {
    -2: FactorizedReduce4,
    -1: FactorizedReduce2,
    0 : ReLUConvBN,
    1 : FactorizedIncrease2,
    2 : FactorizedIncrease4
}

class Cell(nn.Module):
    def __init__(self, steps, C_prev_prev, C_prev, C, state_prev_prev, state_prev, state):
        # 'state'取值1,2,3,4中的某个，分别对应4,8,16,32
        super(Cell, self).__init__()
        self.steps = steps

        if state - state_prev_prev != 0:
            self.pre0 = DIFF_STATE_OPS[state - state_prev_prev](C_prev_prev, C, affine=False, track_stats=False)
        else:
            self.pre0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False, track_stats=False)
        if state - state_prev != 0:
            self.pre1 = DIFF_STATE_OPS[state - state_prev](C_prev, C, affine=False, track_stats=False)
        else:
            self.pre1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False, track_stats=False)

        self._ops = nn.ModuleList()
        for i_b in range(self.steps):
            for i_n in range(2 + i_b):
                self._ops.append(Node(C, 1))

    def forward(self, s0, s1, w_alpha):
        stats = [self.pre0(s0), self.pre1(s1)]
        s_n, e_n = 0, 0
        for i_b in range(self.steps):
            s_n, e_n = e_n, e_n+i_b+2    
            stats.append(sum(self._ops[i_n](stats[i_n-s_n], w_alpha[i_n]) for i_n in range(s_n, e_n)))

        return torch.cat(stats[-self.steps:], dim=1)    # concat the last multiplier blocks's output

class Network(nn.Module):
    def __init__(self, num_classes, num_layers, multiplier=8, steps=5):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.layers = num_layers
        self.steps = steps
        self.multiplier = multiplier
        self.channels = [int(self.steps * self.multiplier * math.pow(2, i + 1) / 4) for i in range(5)]
        self.stem1 = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0])
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[1])
        )

        self.layers_modulelist = {}
        for i_l in range(self.layers):
            if i_l == 0:
                self.layers_modulelist[i_l] = nn.ModuleList([Cell(self.steps, self.channels[0], self.channels[1], self.channels[1], -1, 0, 0),
                                                            Cell(self.steps, self.channels[0], self.channels[1], self.channels[2], -1, 0, 1)]
                                                            )
            if i_l == 1:
                self.layers_modulelist[i_l] = nn.ModuleList([Cell(self.steps, self.channels[1], self.channels[1], self.channels[1], 0, 0, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[1], 0, 1, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[1], self.channels[2], 0, 0, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[2], 0, 1, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[3], 0, 1, 2)]
                                                            )
            if i_l == 2:
                self.layers_modulelist[i_l] = nn.ModuleList([Cell(self.steps, self.channels[1], self.channels[1], self.channels[1], 0, 0, 0),
                                                            Cell(self.steps, self.channels[2], self.channels[1], self.channels[1], 1, 0, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[1], 0, 1, 0),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[0], 1, 1, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[1], self.channels[2], 0, 0, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[0], self.channels[2], 1, 0, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[2], 0, 1, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[2], 1, 1, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[2], 1, 2, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[3], 0, 1, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[3], 1, 1, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[3], 1, 2, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[4], 1, 2, 3)]
                                                            )
            if i_l == 3:
                self.layers_modulelist[i_l] = nn.ModuleList([Cell(self.steps, self.channels[1], self.channels[1], self.channels[1], 0, 0, 0),
                                                            Cell(self.steps, self.channels[2], self.channels[1], self.channels[1], 1, 0, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[1], 0, 1, 0),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[0], 1, 1, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[1], self.channels[2], 2, 1, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[1], self.channels[2], 0, 0, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[1], self.channels[2], 1, 0, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[2], 0, 1, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[2], 1, 1, 1),
                                                            Cell(self.steps, self.channels[3], self.channels[2], self.channels[2], 2, 1, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[2], 1, 2, 1),
                                                            Cell(self.steps, self.channels[3], self.channels[3], self.channels[2], 2, 2, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[3], 0, 1, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[3], 1, 1, 2),
                                                            Cell(self.steps, self.channels[3], self.channels[2], self.channels[3], 2, 1, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[3], 1, 2, 2),
                                                            Cell(self.steps, self.channels[3], self.channels[3], self.channels[3], 2, 2, 2),
                                                            Cell(self.steps, self.channels[3], self.channels[4], self.channels[3], 2, 3, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[4], 1, 2, 3),
                                                            Cell(self.steps, self.channels[3], self.channels[3], self.channels[4], 2, 2, 3),
                                                            Cell(self.steps, self.channels[3], self.channels[4], self.channels[4], 2, 3, 3)]
                                                            )
            else:
                self.layers_modulelist[i_l] = nn.ModuleList([Cell(self.steps, self.channels[1], self.channels[1], self.channels[1], 0, 0, 0),
                                                            Cell(self.steps, self.channels[2], self.channels[1], self.channels[1], 1, 0, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[1], 0, 1, 0),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[0], 1, 1, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[1], self.channels[2], 2, 1, 0),
                                                            Cell(self.steps, self.channels[1], self.channels[1], self.channels[2], 0, 0, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[1], self.channels[2], 1, 0, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[2], 0, 1, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[2], 1, 1, 1),
                                                            Cell(self.steps, self.channels[3], self.channels[2], self.channels[2], 2, 1, 1),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[2], 1, 2, 1),
                                                            Cell(self.steps, self.channels[3], self.channels[3], self.channels[2], 2, 2, 1),
                                                            Cell(self.steps, self.channels[4], self.channels[3], self.channels[2], 3, 2, 1),
                                                            Cell(self.steps, self.channels[1], self.channels[2], self.channels[3], 0, 1, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[2], self.channels[3], 1, 1, 2),
                                                            Cell(self.steps, self.channels[3], self.channels[2], self.channels[3], 2, 1, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[3], 1, 2, 2),
                                                            Cell(self.steps, self.channels[3], self.channels[3], self.channels[3], 2, 2, 2),
                                                            Cell(self.steps, self.channels[4], self.channels[3], self.channels[3], 3, 2, 2),
                                                            Cell(self.steps, self.channels[3], self.channels[4], self.channels[3], 2, 3, 2),
                                                            Cell(self.steps, self.channels[4], self.channels[4], self.channels[3], 3, 3, 2),
                                                            Cell(self.steps, self.channels[2], self.channels[3], self.channels[4], 1, 2, 3),
                                                            Cell(self.steps, self.channels[3], self.channels[3], self.channels[4], 2, 2, 3),
                                                            Cell(self.steps, self.channels[4], self.channels[3], self.channels[4], 3, 2, 3),
                                                            Cell(self.steps, self.channels[3], self.channels[4], self.channels[4], 2, 3, 3),
                                                            Cell(self.steps, self.channels[4], self.channels[4], self.channels[4], 3, 3, 3)]
                                                            )

        self._initialize_alphas()
        self._initialize_betas()
        self.ASPP0 = ASPP(self.channels[1], 256, 3, 1, 24, affine=False, track_stats=False)
        self.ASPP1 = ASPP(self.channels[2], 256, 3, 1, 12, affine=False, track_stats=False)
        self.ASPP2 = ASPP(self.channels[3], 256, 3, 1, 6 , affine=False, track_stats=False)
        self.ASPP3 = ASPP(self.channels[4], 256, 3, 1, 3 , affine=False, track_stats=False)
        self.final_conv = nn.Conv2d(1024, num_classes, 1, bias=False)

    def forward(self, x):
        s0 = [self.stem1(x)]
        s1 = [self.stem2(s0)]
        alpha_sign = 'w_alpha'
        beta_sign = 'w_beta'
        w_alpha = [F.softmax(p[1], 1) for p in self.named_parameters() if p[0].find(alpha_sign) > 0]
        w_beta = [F.softmax(p[1], 0) for p in self.named_parameters() if p[0].find(beta_sign) > 0]

        for i_l in range(self.layers):
            if i_l == 0:
                s_new_0 = self.layers_modulelist[i_l][0](s0[0], s1[0], w_alpha[0]) * w_beta[0][0]
                s_new_1 = self.layers_modulelist[i_l][1](s0[0], s1[0], w_alpha[1]) * w_beta[0][1]
                s0, s1 = s1, [s_new_0, s_new_1]
            elif i_l == 1:
                s_new_0 = self.layers_modulelist[i_l][0](s0[0], s1[0], w_alpha[0]) * w_beta[0][0] * w_beta[1][0] + \
                          self.layers_modulelist[i_l][1](s0[0], s1[1], w_alpha[0]) * w_beta[0][1] * w_beta[2][0]
                s_new_1 = self.layers_modulelist[i_l][2](s0[0], s1[0], w_alpha[1]) * w_beta[0][0] * w_beta[1][1] + \
                          self.layers_modulelist[i_l][3](s0[0], s1[1], w_alpha[1]) * w_beta[0][1] * w_beta[2][1]
                s_new_2 = self.layers_modulelist[i_l][4](s0[0], s1[1], w_alpha[2]) * w_beta[0][1] * w_beta[2][2]
                s0, s1 = s1, [s_new_0, s_new_1, s_new_2]
            elif i_l == 2:
                s_new_0 = self.layers_modulelist[i_l][0] (s0[0], s1[0], w_alpha[0]) * w_beta[1][0] * w_beta[3][0] + \
                          self.layers_modulelist[i_l][1] (s0[1], s1[0], w_alpha[0]) * w_beta[2][0] * w_beta[3][0] + \
                          self.layers_modulelist[i_l][2] (s0[0], s1[1], w_alpha[0]) * w_beta[1][1] * w_beta[4][0] + \
                          self.layers_modulelist[i_l][3] (s0[1], s1[1], w_alpha[0]) * w_beta[2][1] * w_beta[4][0]
                s_new_1 = self.layers_modulelist[i_l][4] (s0[0], s1[0], w_alpha[1]) * w_beta[1][0] * w_beta[3][1] + \
                          self.layers_modulelist[i_l][5] (s0[1], s1[0], w_alpha[1]) * w_beta[2][0] * w_beta[3][1] + \
                          self.layers_modulelist[i_l][6] (s0[0], s1[1], w_alpha[1]) * w_beta[1][1] * w_beta[4][1] + \
                          self.layers_modulelist[i_l][7] (s0[1], s1[1], w_alpha[1]) * w_beta[2][1] * w_beta[4][1] + \
                          self.layers_modulelist[i_l][8] (s0[1], s1[2], w_alpha[1]) * w_beta[2][2] * w_beta[5][0]
                s_new_2 = self.layers_modulelist[i_l][9] (s0[0], s1[1], w_alpha[2]) * w_beta[1][1] * w_beta[4][2] + \
                          self.layers_modulelist[i_l][10](s0[1], s1[1], w_alpha[2]) * w_beta[2][1] * w_beta[4][2] + \
                          self.layers_modulelist[i_l][11](s0[1], s1[2], w_alpha[2]) * w_beta[2][2] * w_beta[5][1]
                s_new_3 = self.layers_modulelist[i_l][12](s0[1], s1[2], w_alpha[3]) * w_beta[2][2] * w_beta[5][2]
                s0, s1 = s1, [s_new_0, s_new_1, s_new_2, s_new_3]
            elif i_l == 3:
                s_new_0 = self.layers_modulelist[i_l][0] (s0[0], s1[0], w_alpha[0]) * w_beta[3][0] * w_beta[6][0] + \
                          self.layers_modulelist[i_l][1] (s0[1], s1[0], w_alpha[0]) * w_beta[4][0] * w_beta[6][0] + \
                          self.layers_modulelist[i_l][2] (s0[0], s1[1], w_alpha[0]) * w_beta[3][1] * w_beta[7][0] + \
                          self.layers_modulelist[i_l][3] (s0[1], s1[1], w_alpha[0]) * w_beta[4][1] * w_beta[7][0] + \
                          self.layers_modulelist[i_l][4] (s0[2], s1[1], w_alpha[0]) * w_beta[5][0] * w_beta[7][0]
                s_new_1 = self.layers_modulelist[i_l][5] (s0[0], s1[0], w_alpha[1]) * w_beta[3][0] * w_beta[6][1] + \
                          self.layers_modulelist[i_l][6] (s0[1], s1[0], w_alpha[1]) * w_beta[4][0] * w_beta[6][1] + \
                          self.layers_modulelist[i_l][7] (s0[0], s1[1], w_alpha[1]) * w_beta[3][1] * w_beta[7][1] + \
                          self.layers_modulelist[i_l][8] (s0[1], s1[1], w_alpha[1]) * w_beta[4][1] * w_beta[7][1] + \
                          self.layers_modulelist[i_l][9] (s0[2], s1[1], w_alpha[1]) * w_beta[5][0] * w_beta[7][1] + \
                          self.layers_modulelist[i_l][10](s0[1], s1[2], w_alpha[1]) * w_beta[4][2] * w_beta[8][0] + \
                          self.layers_modulelist[i_l][11](s0[2], s1[2], w_alpha[1]) * w_beta[5][1] * w_beta[8][0]
                s_new_2 = self.layers_modulelist[i_l][12](s0[0], s1[1], w_alpha[2]) * w_beta[3][1] * w_beta[7][2] + \
                          self.layers_modulelist[i_l][13](s0[1], s1[1], w_alpha[2]) * w_beta[4][1] * w_beta[7][2] + \
                          self.layers_modulelist[i_l][14](s0[2], s1[1], w_alpha[2]) * w_beta[5][0] * w_beta[7][2] + \
                          self.layers_modulelist[i_l][15](s0[1], s1[2], w_alpha[2]) * w_beta[4][2] * w_beta[8][1] + \
                          self.layers_modulelist[i_l][16](s0[2], s1[2], w_alpha[2]) * w_beta[5][1] * w_beta[8][1] + \
                          self.layers_modulelist[i_l][17](s0[2], s1[3], w_alpha[2]) * w_beta[5][2] * w_beta[9][0]
                s_new_3 = self.layers_modulelist[i_l][18](s0[1], s1[2], w_alpha[3]) * w_beta[4][2] * w_beta[8][2] + \
                          self.layers_modulelist[i_l][19](s0[2], s1[2], w_alpha[3]) * w_beta[5][1] * w_beta[8][2] + \
                          self.layers_modulelist[i_l][20](s0[2], s1[3], w_alpha[3]) * w_beta[5][2] * w_beta[9][1]
                s0, s1 = s1, [s_new_0, s_new_1, s_new_2, s_new_3]
            else:
                s_new_0 = self.layers_modulelist[i_l][0] (s0[0], s1[0], w_alpha[0]) * w_beta[(i_l-2)*4-2][0] * w_beta[(i_l-1)*4-2][0] + \
                          self.layers_modulelist[i_l][1] (s0[1], s1[0], w_alpha[0]) * w_beta[(i_l-2)*4-1][0] * w_beta[(i_l-1)*4-2][0] + \
                          self.layers_modulelist[i_l][2] (s0[0], s1[1], w_alpha[0]) * w_beta[(i_l-2)*4-2][1] * w_beta[(i_l-1)*4-1][0] + \
                          self.layers_modulelist[i_l][3] (s0[1], s1[1], w_alpha[0]) * w_beta[(i_l-2)*4-1][1] * w_beta[(i_l-1)*4-1][0] + \
                          self.layers_modulelist[i_l][4] (s0[2], s1[1], w_alpha[0]) * w_beta[(i_l-2)*4]  [0] * w_beta[(i_l-1)*4-1][0]
                s_new_1 = self.layers_modulelist[i_l][5] (s0[0], s1[0], w_alpha[1]) * w_beta[(i_l-2)*4-2][0] * w_beta[(i_l-1)*4-2][1] + \
                          self.layers_modulelist[i_l][6] (s0[1], s1[0], w_alpha[1]) * w_beta[(i_l-2)*4-1][0] * w_beta[(i_l-1)*4-2][1] + \
                          self.layers_modulelist[i_l][7] (s0[0], s1[1], w_alpha[1]) * w_beta[(i_l-2)*4-2][1] * w_beta[(i_l-1)*4-1][1] + \
                          self.layers_modulelist[i_l][8] (s0[1], s1[1], w_alpha[1]) * w_beta[(i_l-2)*4-1][1] * w_beta[(i_l-1)*4-1][1] + \
                          self.layers_modulelist[i_l][9] (s0[2], s1[1], w_alpha[1]) * w_beta[(i_l-2)*4]  [0] * w_beta[(i_l-1)*4-1][1] + \
                          self.layers_modulelist[i_l][10](s0[1], s1[2], w_alpha[1]) * w_beta[(i_l-2)*4-1][2] * w_beta[(i_l-1)*4]  [0] + \
                          self.layers_modulelist[i_l][11](s0[2], s1[2], w_alpha[1]) * w_beta[(i_l-2)*4]  [1] * w_beta[(i_l-1)*4]  [0] + \
                          self.layers_modulelist[i_l][12](s0[3], s1[2], w_alpha[1]) * w_beta[(i_l-2)*4+1][0] * w_beta[(i_l-1)*4]  [0]
                s_new_2 = self.layers_modulelist[i_l][13](s0[0], s1[1], w_alpha[2]) * w_beta[(i_l-2)*4-2][1] * w_beta[(i_l-1)*4-1][2] + \
                          self.layers_modulelist[i_l][14](s0[1], s1[1], w_alpha[2]) * w_beta[(i_l-2)*4-1][1] * w_beta[(i_l-1)*4-1][2] + \
                          self.layers_modulelist[i_l][15](s0[2], s1[1], w_alpha[2]) * w_beta[(i_l-2)*4]  [0] * w_beta[(i_l-1)*4-1][2] + \
                          self.layers_modulelist[i_l][16](s0[1], s1[2], w_alpha[2]) * w_beta[(i_l-2)*4-1][2] * w_beta[(i_l-1)*4]  [1] + \
                          self.layers_modulelist[i_l][17](s0[2], s1[2], w_alpha[2]) * w_beta[(i_l-2)*4]  [1] * w_beta[(i_l-1)*4]  [1] + \
                          self.layers_modulelist[i_l][18](s0[3], s1[2], w_alpha[2]) * w_beta[(i_l-2)*4+1][0] * w_beta[(i_l-1)*4]  [1] + \
                          self.layers_modulelist[i_l][19](s0[2], s1[3], w_alpha[2]) * w_beta[(i_l-2)*4]  [2] * w_beta[(i_l-1)*4+1][0] + \
                          self.layers_modulelist[i_l][20](s0[3], s1[3], w_alpha[2]) * w_beta[(i_l-2)*4+1][1] * w_beta[(i_l-1)*4+1][0]
                s_new_3 = self.layers_modulelist[i_l][21](s0[1], s1[2], w_alpha[3]) * w_beta[(i_l-2)*4-1][2] * w_beta[(i_l-1)*4]  [2] + \
                          self.layers_modulelist[i_l][22](s0[2], s1[2], w_alpha[3]) * w_beta[(i_l-2)*4]  [1] * w_beta[(i_l-1)*4]  [2] + \
                          self.layers_modulelist[i_l][23](s0[3], s1[2], w_alpha[3]) * w_beta[(i_l-2)*4+1][0] * w_beta[(i_l-1)*4]  [2] + \
                          self.layers_modulelist[i_l][24](s0[2], s1[3], w_alpha[3]) * w_beta[(i_l-2)*4]  [2] * w_beta[(i_l-1)*4+1][1] + \
                          self.layers_modulelist[i_l][25](s0[3], s1[3], w_alpha[3]) * w_beta[(i_l-2)*4+1][1] * w_beta[(i_l-1)*4+1][1]
                s0, s1 = s1, [s_new_0, s_new_1, s_new_2, s_new_3]
        x1 = F.interpolate(self.ASPP0(s1[0]), size=x.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.ASPP0(s1[1]), size=x.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.ASPP0(s1[2]), size=x.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.ASPP0(s1[3]), size=x.size()[2:], mode='bilinear', align_corners=True)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.final_conv(out)

    def _initialize_alphas(self):
        c_op   = len(PRIMITIVES)
        c_node = sum(i_b + 2 for i_b in range(self.steps))
        for i_s in range(4):
            self.register_parameter('w_alpha{}'.format(i_s), nn.Parameter(torch.rand((c_node, c_op))))

    def _initialize_betas(self):
        for i_l in range(self.layers):
            if i_l == 0:
                self.register_parameter('w_beta{}_0'.format(i_l), nn.Parameter(torch.rand(2)))
            elif i_l == 1:
                self.register_parameter('w_beta{}_0'.format(i_l), nn.Parameter(torch.rand(2)))
                self.register_parameter('w_beta{}_1'.format(i_l), nn.Parameter(torch.rand(3)))
            elif i_l == 2:
                self.register_parameter('w_beta{}_0'.format(i_l), nn.Parameter(torch.rand(2)))
                self.register_parameter('w_beta{}_1'.format(i_l), nn.Parameter(torch.rand(3)))
                self.register_parameter('w_beta{}_2'.format(i_l), nn.Parameter(torch.rand(3)))
            else:
                self.register_parameter('w_beta{}_0'.format(i_l), nn.Parameter(torch.rand(2)))
                self.register_parameter('w_beta{}_1'.format(i_l), nn.Parameter(torch.rand(3)))
                self.register_parameter('w_beta{}_2'.format(i_l), nn.Parameter(torch.rand(3)))
                self.register_parameter('w_beta{}_3'.format(i_l), nn.Parameter(torch.rand(2)))

if __name__ == '__main__':
    model = Network(21, 12)
