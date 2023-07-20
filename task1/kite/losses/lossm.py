import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .miou import *
def get_mloss(name='di', weight=None):
	seperator = '#'*18
	if weight is not None:
		weight = torch.FloatTensor(weight)
		
	if name=='di':
		print(seperator+'MDice')
		return MDiceLoss(bi=False)
	elif name=='d2':
		print(seperator+'MDice')
		return MDiceLoss(bi=True)
	else:
		return nn.CrossEntropyLoss(weight=weight)
		