# -*- coding: utf-8 -*- 
import torch
from torch import nn
from .utils import *
from .losses import *
from .optims import *

import os, glob, sys, time
from torch.optim import lr_scheduler
torch.set_printoptions(precision=4)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"#

def setup_seed(seed):# 设置随机数种子
	random.seed(seed)
	np.random.seed(seed)

	torch.manual_seed(seed)
	torch.random.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = True

class KiteBack(object):
	lossItem = 0
	device = torch.device('cpu')
	def __init__(self, model, dataset, root=None, **args):
		super(KiteBack, self).__init__()
		print('*'*32, 'Keras Backend Information!')
		self.model = model

		#	设置路径:	清晨morning, 下午afternoon, 傍晚evening, 深夜night(用ABCD代表一天中的次序)
		self.root = 'exp_tcct-bp'
		if not os.path.exists(self.root):
			os.makedirs(self.root)
		self.dataset = dataset
		print('\tFolder for experiment:', self.root)

		#	模型权重加载，断点续练
		if not self.weights_load('los'):
			print('weights_init_kaiming')
			# init_weights(self.model, init_type='kaiming')
		print('\tParams model:',sum(p.numel() for p in self.model.parameters() if p.requires_grad))

		#	回调函数

	def cuda(self, m):
		return m.to(self.device)

	def islrLowerThan(self, thresh=1e-5):
		return self.optimG.param_groups[0]['lr'] < thresh

	def grad_dump(self, epoch):
		# tarGrad = {'epoch':epoch, 'loss':self.lossName, 'lr':self.schedG.get_last_lr()}
		tarGrad = {'epoch':epoch, 'loss':self.lossName, 'lr':self.optimG.param_groups[0]['lr']}
		torch.save(tarGrad, self.checkpoint_grad)

	coff_ds = 0.5
	def grad_calc(self, outs, true, ds=True, criterion=None):
		losSum = 0
		if isinstance(outs, (list, tuple)):# or isinstance(outs):
			if ds:
				for i in range(len(outs)-1,0,-1):#第一个元素尺寸最大，起止索引莫写错了啊啊啊
					loss = criterion(outs[i], true)
					losSum = losSum + loss*self.args.coff_ds
			outs = outs[0]
		# print(outs.shape, true.shape)
		loss = criterion(outs, true)
		losSum = losSum + loss
		return losSum
	
	#	模型权重加载、初始化、打印
	def weights_load(self, mode, desc=True):
		if not mode.endswith('.pt'):
			path = os.path.join(self.root, mode+'.pt')#self.paths.get(mode, mode)#返回完全路径或者mode
			# path = glob.glob(self.root+'/*'+mode+'*.pt')[0]
		try:
			pt = torch.load(path, map_location=self.device)
			self.model.load_state_dict(pt, strict=False)
			if desc:print('\nLoad weight:', path)
			return True
		except:
			print('\nLoad weight wrong:', path)
			return False

	def weights_desc(self, key='my'):#, self.schedG.get_lr()[0]
		# print('Learing Rate:', self.optimG.param_groups[0]['lr'])
		for n,m in self.model.named_parameters():
			if n.__contains__(key):
				print(n,m.detach().cpu().numpy())
	
	def remove_pths(self, flag_ignore='los'):
		for path in glob.glob(self.root+'/*.pt'):
			# path = self.root+'/'+key+'.pt'
			if flag_ignore not in path:
				os.remove(path)

	#	设置：超参数、显卡
	def set_superes(self, loss='ce', lr=0.01, wd=2e-4, **args): 
		print('Setting super parameters!!!')
		#	参数设置：反向传播、断点训练 
		self.checkpoint_grad = os.path.join(self.root, 'params.tar')

		if os.path.isfile(self.checkpoint_grad):
			try:
				tarGrad = torch.load(self.checkpoint_grad)
				epoch = tarGrad['epoch']
				loss = tarGrad['loss']
				lr = tarGrad['lr']
				print('Load Super params for Gradients, lr:{}, los:{}'.format(lr, loss))
			except:
				epoch = 0
				print('Load Super params Failed!, lr:{}, los:{}'.format(lr, loss))
		else:
			epoch = 0
			print('Init Super params for Gradients, lr:{}, los:{}'.format(lr, loss))

		self.epoch = epoch
		print('$'*32, 'start-epoch:{}'.format(self.epoch))
		self.lossName = loss
		# self.criterion = get_loss(loss)

		params = filter(lambda p:p.requires_grad, self.model.parameters())
		self.optimG = torch.optim.AdamW(params=params, lr=lr, weight_decay=wd)
		self.schedG = lr_scheduler.CyclicLR(self.optimG, base_lr=1e-6, max_lr=1e-4, cycle_momentum=False, step_size_up=4, step_size_down=60)

	def set_backend(self, gpu='0', parallel=False, **args):
		print('Setting backend for Pytorch!!!')
		if torch.cuda.is_available():
			self.device = torch.device('cuda:0')  
			torch.cuda.empty_cache()

			print('Using GPU:')
			self.model = self.model.to(self.device) 

		self.criterion = get_loss(self.args.los).to(self.device)
		