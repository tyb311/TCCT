
import torch
from torch import nn
from torch.nn import functional as F


import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'rainbow'

# https://zhuanlan.zhihu.com/p/512117483
import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号’-'显示为方块的问题
# fpath = os.path.join(plt.rcParams['datapath'], "fonts/ttf/cmr10.ttf")
# fpath = r"C:\Python38\Lib\site-packages\matplotlib\mpl-data\fonts/ttf/cmr10.ttf"
# prop = mpl.font_manager.FontProperties(fname=fpath, size=8)
# strech，拉伸，相当于word中的字体加宽
# import matplotlib.font_manager as fm
# prop = fm.FontProperties(family='FangSong',size=24, stretch=0)
# prop = fm.FontProperties(family='Mistral',size='xx-large',stretch=1000, weight='bold')


def points_selection_bins(feat, prob, true, card=512, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[1]
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	prob = prob[true>.5].view(-1)
	# print('ft:', feat.shape, feat.min().item(), feat.max().item())
	# with torch.no_grad():
	seq, idx = torch.sort(prob, dim=-1, descending=True)#降序排列
	# print('seq:', seq)	#seq: tensor([0.2426, 0.2354, 0.2335,  ..., 0.0633, 0.0612, 0.0608])
	# print('sort:', seq.shape, idx.shape, feat.shape, prob.shape)
	bins = 32
	N = feat.shape[0]//bins
	fs = []
	for i in range(bins):
		ni = N*i
		nj = ni+N
		f = torch.index_select(feat, dim=0, index=idx[ni:nj])
		# print('lh:', l.shape, h.shape, N, i)
		# print(h.shape, l.shape)
		f = f.mean(dim=0, keepdim=True)
		fs.append(f)
	f = torch.cat(fs, dim=0)
	# print('lh:', l.shape, h.shape, l.max().item(), h.max().item())
	# print(prob[idx[:card]].view(-1)[:9])
	# print(prob[idx[-card:]].view(-1)[:9])
	return f

class FeatConSuper(nn.Module):
	def __init__(self, con='cos', mode='bins', *args):
		super(FeatConSuper, self).__init__()
		self.__name__ = con
		self.mse = nn.MSELoss(reduction='mean')
		self.con = con

			# self.register_buffer('off_diag', 1-torch.eye(dim, dtype=torch.float32))
		self.forward = self.cosinesim
		self.func = points_selection_bins

	def cosinesim(self, q, k):#负太多了
		# return - F.cosine_similarity(q, k, dim=-1).mean()
		# q = F.normalize(q, dim=-1)
		# k = F.normalize(k, dim=-1)
		return - torch.einsum('nc,kc->nk', [q, k]).mean() / q.shape[-1]

	def foreach_loss(self, fts, gts):
		losFor = 0
		for i,ft in enumerate(fts):
			for j,gt in enumerate(gts):
				if i==j:
					losFor = losFor + self.forward(ft,gt)
				# elif j>i:
				# 	# losFor = losFor - self.forward(ft,gt)
				# 	losFor = losFor + F.cosine_similarity(ft, gt).mean()
		# fts = torch.cat(fts, dim=0)
		# gts = torch.cat(gts, dim=0)
		return losFor# - F.cosine_similarity(fts, gts).mean()
	
	def select1(self, feat, pred, true, mask=None, ksize=5, card_select=16):
		# print(feat.shape, pred.shape, true.shape)
		assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		dim_latent = feat.shape[1]
		# reshape embeddings
		feat = feat.permute(0,2,3,1).reshape(-1, dim_latent)
		# feat = F.normalize(feat, p=2, dim=-1)
		true = true.float().round()
		# true = F.max_pool2d(true, kernel_size=(3,3), stride=1, padding=1)
		# print('maxpool:', true.shape)
		num_smp = card_select*true.shape[0]
		ft = self.func(feat,   pred, true, card=num_smp)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		return ft
		# print(mode)
		# return torch.cat([fh, fl], dim=0)



