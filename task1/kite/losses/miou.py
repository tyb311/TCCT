import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable#,Function
from torch.nn import CrossEntropyLoss
# from torch.nn.modules.loss import CrossEntroyLoss
class OneHot(nn.Module):
	"""docstring for OneHot"""
	def __init__(self, nb_class):
		super(OneHot, self).__init__()
		self.nb_class = nb_class
	def forward(self, x):
		# print('OneHot:', x.shape, x.max().item(), self.nb_class)
		return F.one_hot(x, self.nb_class)
		# n = x.size()[0]
		# h, w = x.size()[-2:]
		# zeros = torch.zeros(n, self.nb_class, h, w, device=x.device)
		# one_hot = zeros.scatter_(1, x.view(n, 1, h, w), 1)
		# return one_hot

#start#
class MIouLoss(nn.Module):
	'''two classes'''
	def __init__(self, nb_class=4):
		super(MIouLoss, self).__init__()
		self.nb_class = nb_class

	@staticmethod
	def score(pr, gt, smooth=1):
		# pr = F.softmax(pr, dim=1).round().long()
		pr = pr.reshape(pr.shape[0],-1)
		gt = gt.reshape(pr.shape[0],-1)
		inter = torch.sum(pr * gt, dim=-1)
		union = torch.sum(pr, dim=-1) + torch.sum(gt, dim=-1) - inter + smooth
		# print('iou:', inter.shape, union.shape)

		score = torch.mean((inter + smooth) / union)
		return score
		
	@staticmethod
	def scorem(pr, gt, start_idx=0):
		# pr = F.softmax(pr, dim=1).round().long()
		score = sum([MIouLoss.score(pr[:,i:i+1],gt[:,i:i+1]) for i in range(start_idx,pr.shape[1])])
		return score/(pr.shape[1]-start_idx)

	def forward(self, pr, gt, smooth=1e-6):
		batch_size = pr.size(0)
		nb_class = pr.shape[1]

		if gt.shape!=pr.shape:
			gt = F.one_hot(gt, pr.shape[1]).contiguous().permute(0,3,1,2)
		gt = gt.view(batch_size, nb_class, -1)
		pr = F.softmax(pr, dim=1).view(batch_size, nb_class, -1)
		# print('pr gt:', pr.shape, gt.shape)

		inter = torch.sum(pr * gt, dim=-1)
		union = torch.sum(pr, dim=-1) + torch.sum(gt, dim=-1) -inter + smooth
		# print('iou:', inter.shape, union.shape)

		score = torch.sum(inter / union)
		score = 1.0 - score / (float(batch_size) * float(nb_class))
		return score

class MDiceLoss(nn.Module):
	def __init__(self, nb_class=2, bi=False):
		super(MDiceLoss, self).__init__()
		self.func = self.dice2 if bi else self.dice

	@staticmethod
	def score(pr, gt, smooth=1):
		# print('score:', pr.shape, gt.shape, pr.max().item(), gt.max().item())
		# pr = F.softmax(pr, dim=1).round().long()
		pr = pr.reshape(pr.shape[0],-1)
		gt = gt.reshape(pr.shape[0],-1)
		inter = torch.sum(pr * gt, dim=-1)
		union = torch.sum(pr, -1) + torch.sum(gt, -1) + smooth
		# print('iou:', inter.shape, union.shape)

		score = (2*inter + smooth) / union
		return score.mean()
		
	@staticmethod
	def scores(pr, gt):
		# pr = F.softmax(pr, dim=1).round().long()
		return [MDiceLoss.score(pr[:,i:i+1],gt[:,i:i+1]).cpu().item() for i in range(pr.shape[1])]
		
	@staticmethod
	def scorem(pr, gt, start_idx=0):
		# pr = F.softmax(pr, dim=1).round().long()
		score = sum([MDiceLoss.score(pr[:,i:i+1],gt[:,i:i+1]) for i in range(start_idx,pr.shape[1])])
		return score/(pr.shape[1]-start_idx)

	def forward(self, pr, gt):
		batch_size = pr.size(0)
		nb_class = pr.shape[1]
		# print('MDiceLoss:', pr.shape, gt.shape)
		if gt.shape!=pr.shape:
			gt = F.one_hot(gt, pr.shape[1]).contiguous().permute(0,3,1,2)
		gt = gt.view(batch_size, nb_class, -1)
		pr = F.softmax(pr, dim=1).view(batch_size, nb_class, -1)
		# gt = gt[:, self.class_ids, :]
		return self.func(pr, gt)

	def dice2(self, pr, gt):
		return self.dice(pr, gt) + self.dice(1-pr, 1-gt)

	def dice(self, pr, gt, smooth=1e-6):
		batch_size = pr.size(0)
		nb_class = pr.shape[1]

		inter = torch.sum(pr * gt, 2) + smooth
		union = torch.sum(pr, 2) + torch.sum(gt, 2) + smooth

		score = torch.sum(2.0 * inter / union)
		score = 1.0 - score / (float(batch_size) * float(nb_class))

		return score
#end#



# class MDiceLoss(nn.Module):
#     '''multy classes'''
#     def __init__(self, nb_class=2, bi=False, ch=False):
#         super(MDiceLoss, self).__init__()
#         self.nb_class = nb_class
#         self.func = self.dice2 if bi else self.dice
#         if ch:
#             self.func = self.ch_dice

#     def forward(self, pr, gt):
#         batch_size = pr.size(0)
#         pr = F.softmax(pr, dim=1)
#         pr = pr.view(batch_size, self.nb_class, -1).sigmoid()
#         gt = F.one_hot(gt, pr.shape[1]).contiguous().contiguous().view(batch_size, self.nb_class, -1)
#         # gt = gt[:, self.class_ids, :]
#         return self.func(pr, gt)

#     def dice2(self, pr, gt):
#         return self.dice(pr, gt) + self.dice(1-pr, 1-gt)

#     def ch_dice(self, pr, gt, smooth=1e-6):
#         # batch_size = pr.size(0)

#         inter = torch.sum(pr * gt, dim=2) + smooth
#         union = torch.sum(pr, dim=2) + torch.sum(gt, dim=2) + smooth

#         score = 1.0 - 2.0 * torch.mean(inter / union)# / (float(batch_size) * float(self.nb_class))
#         return score

#     def dice(self, pr, gt, smooth=1e-6):
#         # batch_size = pr.size(0)

#         inter = torch.sum(pr * gt) + smooth
#         union = torch.sum(pr) + torch.sum(gt) + smooth

#         score = 1.0 - 2.0 * inter / union# / (float(batch_size) * float(self.nb_class))
#         return score






if __name__ == '__main__':
	NB_CLASS = 7
	BATCH_ES = 3
	y = torch.LongTensor(BATCH_ES,64,64).random_() % (NB_CLASS-1)#  # 4 classes,1x3x3 img
	# y = y.float()
	x = torch.rand(BATCH_ES,NB_CLASS,64,64)#.float()
	print(x.min().item(), x.max().item(), y.min().item(), y.max().item())

	loss = MIouLoss(nb_class=NB_CLASS)(x, y)
	print('MIouLoss:', loss.item())

	loss = MDiceLoss(nb_class=NB_CLASS)(x, y)
	print('MIouLoss:', loss.item())

	loss = nn.CrossEntropyLoss()(x, y)
	print('CrossEntropyLoss:', loss.item())
