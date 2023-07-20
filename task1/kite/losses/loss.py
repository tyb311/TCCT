import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
	__name__ = 'DiceLoss'
	# DSC(A, B) = 2 * |A ^ B | / ( | A|+|B|)
	def __init__(self, bi=False):
		super(DiceLoss, self).__init__()
		self.func = self.dice2 if bi else self.dice
	def forward(self, pr, gt, **args):
		# assert 0<=pr.min().item()<=pr.max().item()<=1, 'prediction is not binary!'
		# if self.bi:
		#     return 2-self.dice(pr,gt)-self.dice(1-pr,1-gt)
		los = 1-self.func(pr, gt)
		return los
	@staticmethod
	def dice2(pr, gt, smooth=1):#self, 
		pr,gt = pr.reshape(-1),gt.reshape(-1)
		inter = (pr*gt).sum()
		union = (pr**2+gt**2).sum()
		return (smooth + 2*inter) / (smooth + union)
	@staticmethod
	def dice(pr, gt, smooth=1):#self, 
		pr,gt = pr.reshape(-1),gt.reshape(-1)
		inter = (pr*gt).sum()
		union = (pr+gt).sum()
		return (smooth + 2*inter) / (smooth + union)
	@staticmethod
	def dicem(pr, gt, smooth=1e-6):
		# pr = F.softmax(pr, dim=1).round().long()
		score = sum([DiceLoss.dice(pr[:,i:i+1],gt[:,i:i+1]) for i in range(pr.shape[1])])
		return score/pr.shape[1]



class IouLoss(nn.Module):
	__name__ = 'IouLoss'
	# IOU(A,B) = |A ^ B| / |A U B|
	def __init__(self, bi=False):
		super(IouLoss, self).__init__()
		self.bi = bi
	def forward(self, pr, gt, **args):
		# if self.bi:
		#     return 2-self.iou(1-pr,1-gt)-self.iou(pr, gt)
		return 1-self.iou(pr, gt)
	@staticmethod
	def iou(pr, gt, smooth=1e-12):
		pr,gt = pr.reshape(-1),gt.reshape(-1)
		inter = (pr*gt).sum()
		union = (pr+gt).sum()-inter
		return (inter+smooth)/(union+smooth)
	@staticmethod
	def miou(pr, gt, ignore_index=0):
		assert 0<=pr.max().item()<=1, 'prediction is not binary!'
		assert 0<=gt.shape[1]<=9, 'gt lesion class is wrong!'
		assert 0<=pr.shape[1]<=9, 'pr lesion class is wrong!'
		mious = []
		for i in range(gt.shape[1]):
			if i==ignore_index:
				continue
			mious.append(IouLoss.iou(pr[:,i], gt[:,i]).item())
		losStr = ','.join(['{:.4f}'.format(it) for it in mious])
		return sum(mious)/len(mious), losStr

class MultiLoss(nn.Module):
	__name__ = 'MultiLoss'
	def __init__(self, losses, weight=None):#[1,1,1,1,1,1,1,1,10,1,1]
		super(MultiLoss, self).__init__()

		self.losses = losses
		if weight is None:
			self.WEIGHT = [1,]*40#torch.Tensor([1,]*19)#.cuda()
		#     self.icpr = nn.CrossEntropyLoss(weight=self.WEIGHT, reduction='mean')
		#     self.pool = nn.AdaptiveMaxPool2d(1)
		else:
			self.WEIGHT = weight

	def forward(self, pr, gt, **args):
		# print('before:', pr.shape, gt.shape, pr.device, gt.device)
		# if pr.min()!=0 or pr.max()!=1:
		pr = torch.softmax(pr, dim=1)
		los = 0

		if gt.shape[1]!=pr.shape[1]:
			gt = F.one_hot(gt.to(pr.device), pr.shape[1]).contiguous().permute(0,3,1,2)
		# print('after:', pr.shape, gt.shape)
		losses = []
		for i in range(gt.shape[1]):
			los = self.losses(pr[:,i:i+1], gt[:,i:i+1])
			losses.append(los)


		los = sum(l*w for l,w in zip(losses, self.WEIGHT))
		return los

def get_loss(loss='di', **args):
	# 交叉熵损失系列(balanced or not)
	if loss=='dice' or loss=='di':
		print(loss, 'DiceLoss()')
		los =  DiceLoss(bi=False)
	else:
		print('MSE')
		los = nn.MSELoss()

	return MultiLoss(los)
	