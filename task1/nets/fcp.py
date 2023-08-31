
import torch
from torch import nn
from torch.nn import functional as F

import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'rainbow'

from sklearn.manifold import TSNE
from tqdm import tqdm



class FeatConPolar(nn.Module):#Uniformly divided hypersphere
	def __init__(self, num_cls=8, num_emb=32, init=True):
		super(FeatConPolar, self).__init__()
		self.num_cls = num_cls
		self.vec_grad = nn.Parameter(torch.rand(num_cls, num_emb), requires_grad=True)
		self.cls_nums = torch.arange(0,num_cls).long()
		print('FeatConPolar-Number&Length:', num_cls, num_emb)
		N = num_cls*(num_cls-1)//2
		print('Constrain vectors to', -1/(num_cls-1))
		self.register_buffer('cos_dist', torch.FloatTensor([-1/(num_cls-1)]*N))
		# self.register_buffer('cos_dist', torch.FloatTensor([-1]*N))

		ies, jes = [], []
		for i in range(0, num_cls):
			ies.append(i)
			jes.append((i+1)%num_cls)
		self.ies = torch.LongTensor(ies)
		self.jes = torch.LongTensor(jes)
		# print(self.ies, self.jes)
	
		if init:
			optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, betas=(0.9, 0.999), weight_decay=2e-4)
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,  
				mode='min', factor=0.7, patience=2, 
				verbose=False, threshold=0.0001, threshold_mode='rel', 
				cooldown=2, min_lr=1e-5, eps=1e-9)

			tbar = tqdm(range(333), desc='\r')
			for i, j in enumerate(tbar):
			# while True:
				optimizer.zero_grad()
				los = self.regular_target(self.vec_grad)
				los.backward()
				optimizer.step()
				scheduler.step(los.item())
				tbar.set_description('los-udh={:.4f}'.format(i, los.item()))
				if los.item()<1e-5:
					print('\n################After {} epochs, Loss for UDH is:{:.4f}'.format(i, los.item()))
					for p in self.parameters():
						p.requires_grad = False
					break
			tbar.close()

		# self.desc()
		self.vec_grad.requires_grad = False
		self.register_buffer('buf_grad', F.normalize(self.vec_grad, p=2, dim=-1).detach())
		print('vec_grad:', self.vec_grad.min().item(), self.vec_grad.max().item())

	def regular_target(self, vec_nd):
		vec_grad = F.normalize(vec_nd, dim=-1)

		losSum = torch.log(torch.exp(vec_grad @ vec_grad.T).mean(dim=-1)).mean()

		return losSum


	def choice(self, pro, i):#特征分组、组间排斥、组内同化
		lab = torch.ones(size=(pro.shape[0],1), dtype=torch.long, device=pro.device).reshape(-1)*i
		vec = torch.index_select(self.buf_grad, dim=0, index=lab)
		return vec
		
'''
空间维度	平分向量	夹角余弦
	1			2		-1
	2			3		-1/2
	3			4		-1/3
	N			N+1		-1/N

假设类别数为N+1，则对应的空间维度设置为大于N比较好（等于N时为等分向量）
'''
