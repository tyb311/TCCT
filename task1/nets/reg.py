
import torch
from torch import nn
from torch.nn import functional as F


from .fcs import *
from .fcp import *


class BaseNet(nn.Module):#Enhanced Three branches [Distance Map Regression] Net
	__name__ = 'base'
	def __init__(self, out_channels=5, num_emb=32):#(32,64,96,128,160)
		super(BaseNet, self).__init__()

		self.filt = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=3,stride=1, padding=1),
			nn.Conv2d(8, num_emb, kernel_size=3, stride=1, padding=1),
		)
		self.out = nn.Conv2d(num_emb, out_channels, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		self.feat = self.filt(x)
		self.feats = [self.feat]*4
		return self.out(self.feat)

def soft_argmax(x, beta=100):
	# https://blog.csdn.net/hxxjxw/article/details/121596524
	# x = x.reshape(x.shape[0], x.shape[1], -1)#[b,c,h,w]
	soft_max = F.softmax(x*beta, dim=1).view(x.shape).clamp(0,1)
	# soft_max = F.gumbel_softmax(x, dim=1, tau=beta)
	idx_weight = torch.arange(start=0, end=x.shape[1]).reshape(1,-1,1,1)
	# print(x.shape, soft_max.shape, idx_weight.shape)
	matmul = soft_max * idx_weight.to(soft_max.device)
	return matmul.sum(dim=1, keepdim=True)


class RegNet(nn.Module):
	__name__ = 'reg'
	tmp={}
	def __init__(self, base, out_channels=5, con='cor', num_emb=32):#(32,64,96,128,160)
		super(RegNet, self).__init__()
		self.base = base
		self.__name__ = base.__name__
		self.out_channels = out_channels

		# self.filt = nn.Sequential(
		# 	nn.Conv2d(out_channels, 8, kernel_size=3,stride=1, padding=1),
		# 	nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
		# 	nn.BatchNorm2d(8),
		# 	nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
		# 	nn.Sigmoid()
		# )

		# self.projector = MlpNorm(num_emb, 32)
		self.fcs = FeatConSuper(con=con)
		self.fcp = FeatConPolar(num_cls=out_channels, num_emb=32, init=False)
		self.lap_epl = nn.Sequential(
			nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1),
			# nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid()
		)
		dim_reg = out_channels-1
		self.lap_reg = nn.Sequential(
			nn.Conv2d(dim_reg, dim_reg, kernel_size=3, stride=1, padding=1, groups=dim_reg),
			nn.Conv2d(dim_reg, dim_reg, kernel_size=3, stride=1, padding=1, groups=dim_reg),
			# nn.Conv2d(dim_reg, dim_reg, kernel_size=3, stride=1, padding=1, groups=dim_reg),
			# nn.Conv2d(dim_reg, 1, kernel_size=3, stride=1, padding=1, groups=1),
		)
		self.lap_map = nn.Sequential(
			nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1),
			nn.BatchNorm2d(1,1),
			nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1),
			nn.Sigmoid()
		)
		self.tau = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32)*100)

	def forward(self, x):
		y = self.base(x)
		return y

	def plot_udh(self, path_save='exps/feat_udh.png'):
		pass

	def regular_udh(self, pred, true, tau=5):
		# feat = self.base.feat#.clone()
		# feat = F.normalize(self.base.feat, dim=1, p=2)#select前norm和select后norm效果可能也不一样
		pred = torch.softmax(pred.detach(), dim=1)
		losCon = 0
		for feat in self.base.feats:
			pros, tgts = [], []
			for i in range(true.shape[1]):
				pro = self.fcs.select1(feat=feat, pred=pred[:,i:i+1], true=true[:,i:i+1])
				# pro = self.projector(pro)#.clamp(-3,3)
				# pro = F.normalize(pro, p=2, dim=1)
				# tgt = pro[0:1].detach().repeat(pro.shape[0],1)
				tgt = self.fcp.choice(pro, i)
				# print(i, pro.shape, tgt.shape)
				pros.append(pro)
				tgts.append(tgt)
			losCon = losCon + self.fcs.foreach_loss(pros, tgts) + F.mse_loss(pro, tgt)
			self.emb_list = pros
			self.tgt_list = tgts
		return losCon

	# loss = FocalLoss()
	loss = nn.MSELoss()
	def regular_reg(self, pred, true, tau=100):
		pred = pred[:,1:]
		true = true[:,1:].float()
		B,C,H,W = pred.shape
		prob_true = F.pad(torch.abs(true[:,:,1:] - true[:,:,:-1]).float(), pad=(0,0,1,0), mode='constant', value=0)
		prob_true = prob_true.sum(dim=1).unsqueeze(1).clamp_max(1)
		pseu_pred = self.lap_reg(pred).abs()
		pseu_true = self.lap_reg(true).abs()

		def sampling_softmax(pred):
			# return F.softmax(pred*self.tau.clamp(1,1000), dim=-2)
			eps = torch.rand_like(pred, device=pred.device)
			log_eps = torch.log(-torch.log(eps))
			# print('eps:', log_eps.min().item(), log_eps.max().item())
			gumbel_pred = pred - log_eps / 2 #self.tau.abs()
			gumbel_pred = F.softmax(gumbel_pred, dim=-2)
			# print('gumbel_pred:', gumbel_pred.shape)
			return gumbel_pred / (1e-6+gumbel_pred.sum(dim=-2).unsqueeze(2))

		pseu_pred = self.lap_map(sampling_softmax(pseu_pred).sum(dim=1).unsqueeze(1))
		pseu_true = self.lap_map(sampling_softmax(pseu_true).sum(dim=1).unsqueeze(1))
		# pseu_pred = self.lap_map(pseu_pred.sum(dim=1).unsqueeze(1))
		# pseu_true = self.lap_map(pseu_true.sum(dim=1).unsqueeze(1))
		# pseu_true = pseu_true.sum(dim=1).unsqueeze(1)
		# plt.subplot(131),plt.imshow(pseu_pred[0,0].data.numpy())
		# plt.subplot(132),plt.imshow(pseu_true[0,0].data.numpy())
		# plt.subplot(133),plt.imshow(prob_true[0,0].data.numpy())
		# plt.show()
		# pseu_pred = F.gumbel_softmax(pred, 1/tau, dim=-2)#总是出现nan
		# print('assert:', pseu_pred.sum(-2).min().item(), pseu_pred.sum(-2).max().item())
		# print('prob_true:', prob_true.shape, prob_true.min().item(), prob_true.max().item())
		# print('pseu_pred:', pseu_pred.shape, pseu_pred.min().item(), pseu_pred.max().item())
		self.tmp['reg_pred'] = pseu_pred[0].unsqueeze(1)
		self.tmp['reg_true'] = prob_true[0].unsqueeze(1)

		# print('idx_weight:', idx_weight.shape, idx_weight.max().item())
		# print('RANGE:', pseu_pred.shape, pseu_pred.max().item(), prob_true.shape, prob_true.max().item())
		idx_weight = torch.arange(0, H).reshape(1,1,-1,1).float().to(pseu_pred.device)
		idxt_weight = idx_weight+torch.rand_like(idx_weight).to(idx_weight.device) - 0.5
		idxp_weight = idx_weight+torch.rand_like(idx_weight).to(idx_weight.device) - 0.5
		edge_true = (pseu_true*idxt_weight).sum(dim=-2)/H
		edge_pred = (pseu_pred*idxp_weight).sum(dim=-2)/H
		# print('edge_pred:', edge_pred.shape, edge_pred.min().item(), edge_pred.max().item())
		# print('edge_true:', edge_true.shape, edge_true.min().item(), edge_true.max().item())
		# return F.l1_loss(edge_pred, edge_true)
		losEdge = self.loss(edge_pred, edge_true.detach()) + self.loss(edge_pred.detach(), edge_true)
		losProb = self.loss(prob_true, pseu_true.softmax(dim=-2)) + self.loss(prob_true, pseu_pred.softmax(dim=-2))#
		return  losEdge + losProb
		# return F.mse_loss(edge_pred, edge_true)
		