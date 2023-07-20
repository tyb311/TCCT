import torch
from torch.autograd import Variable
# 事实证明torch.round、torch.max无法使用反向传播


from .loss import *
