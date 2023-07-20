# -*- coding: utf-8 -*-
from kite.losses.miou import MDiceLoss, MIouLoss

from .loopback import KiteBack, setup_seed
from .losses import *

from tqdm import tqdm
import time

class KiteSeg(KiteBack):
	def __init__(self, args, **_args):
		self.args = args
		super(KiteSeg, self).__init__(**_args)
		self.set_superes(loss=args.los, lr=args.lr)#, wd=args.wd
		self.set_backend(gpu=args.gpu, parallel=args.pl)

		NB_CLASS = self.dataset.out_channels
		self.NB_CLASS = NB_CLASS
		self.criterion.NB_CLASS = NB_CLASS

	def predict(self, img, softmax=True, *args):	# predict outputs from input
		with torch.no_grad():
			img = self.cuda(img)

			pred = self.model(img)#对于TCCT、CCTP效果更好

			if isinstance(pred, (list, tuple)):
				pred = pred[0]
			pred = pred.detach()

			if softmax:
				pred = F.one_hot(torch.argmax(F.softmax(pred,dim=1),dim=1), self.NB_CLASS).permute(0,3,1,2).float()
		return pred

	useValSet = True
	cnt_val = 0
	def fit(self, epochs=169):#现行验证，意义不大，把所有权重都验证要花不少时间
		print('\n', '*'*8, 'Fitting:'+self.root)
		self.callback.save_key(self.model, 'val_iou')
		time_fit_begin = time.time()
		for i in range(self.epoch, epochs):
			time_stamp = time.time()
			# 训练
			losItem = self.train(i)

			self.schedG.step()

			# 验证
			if i%10==0 or (i>0.5*epochs and i%5==0):
				if hasattr(self.model, 'base.feats') or hasattr(self.model, 'base.feat'):
					self.model.regular_udh(self.udh_out, self.udh_lab)

				logs = self.val(epoch=i)	
				if i%30==0:
					self.callback.save_key(self.model, 'e{}'.format(i))
				self.callback.update(logs=logs, model=self.model)

		
			self.grad_dump(i)
			# 早停&训练验证过程加入学习率和损失的tensorboard曲线
			# if self.callback.stop_training and i>0.6*epochs:
			# 	print('Stop Training!!!')
			# 	break
			time_epoch = time.time() - time_stamp
			print('{:03}* {:.2f} mins, left {:.2f} hours to run'.format(i, time_epoch/60, time_epoch/60/60*(epochs-i)))
				
		logTime = '\nRunning {:.2f} hours for {} epochs!'.format((time.time() - time_fit_begin)/60/60, epochs)
		self.weights_desc()
	
	def val(self, epoch=0, flagDebug=False):#GPU 加速评价，所以评价函数也要用PyTorch实现
		torch.set_grad_enabled(False)
		self.model.eval()
		sum_iou = 0
		sum_f1s = 0
		tbar = tqdm(self.dataset.valSet(bs=1)) 
		# tbar = tqdm(self.dataset.testSet(bs=1)) 
		scores=[]
		for i, imgs in enumerate(tbar):
			(img, lab, fov, aux) = self.dataset.parse(imgs)
			img = self.cuda(img)
			lab = self.cuda(lab).long()

			# print('val:', img.shape, lab.shape)
			out = self.predict(img, softmax=True).detach()
			if out.shape[1]!=self.NB_CLASS:
				print("output shape not equals NB_CLASSES!")
				
			lab = F.one_hot(lab, self.NB_CLASS).permute(0,3,1,2)#out.shape[1]
			true = lab.long().detach()
			pred = out.detach()
			
			f1s = MDiceLoss.scorem(pred, true, start_idx=1).cpu().item()
			iou = MIouLoss.scorem(pred, true, start_idx=1).cpu().item()
			score = np.array(MDiceLoss.scores(pred, true)).reshape(-1).astype(np.float32)
			scores.append(score)
			sum_iou += iou
			sum_f1s += f1s

			# print('\r${:03} los={:.4f} iou={:.4f}'.format(i, los, iou), end='')
			tbar.set_description('Val@{:03} iou={:.4f} & f1s={:.4f}'.format(epoch, iou, f1s))
			# scores.append(iou)
			if (self.args.bug or flagDebug) and i>8:
				break

		i = len(tbar)
		tbar.close()
		logs = {'val_iou':sum_iou/i, 'val_f1s':sum_f1s/i}
		scores = np.round(np.stack(scores, axis=0).mean(axis=0), 4)
		print('*SCORES:*', scores, '->', scores[1:].mean())
		return logs

	def train(self, epoch, alpha=.9):
		setup_seed(epoch*311+2023)
		torch.set_grad_enabled(True)
		losItem = 0
		self.model.train()
		tbar = tqdm(self.dataset.trainSet(bs=self.args.bs))
		for i, imgs in enumerate(tbar):
			(img, lab, fov, aux) = self.dataset.parse(imgs)
			img = self.cuda(img)
			lab = self.cuda(lab).long().squeeze()
			# print('train:', img.shape)
			lab = F.one_hot(lab, self.NB_CLASS).permute(0,3,1,2)#out.shape[1]
			
			self.optimG.zero_grad()#避免历史信息和wd对参数的更新
			#	反向传播

			losSum, logStr = self.calc_loss(img, lab)
			# #梯度裁剪
			losSum.backward()
			# torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.1)#clip_value=1.1
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)#（最大范数，L2)
			# torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)#（最大范数，L2)
			self.optimG.step()

			# tbar.set_description('los='+','.join(['%.4f' % los.item() for los in costList]))    
			tbar.set_description(logStr)
			losItem += losSum.item()
			# print('\r#{:03}* los={:.3f}'.format(i, losSum.item()), end='')
			if self.args.bug and i>12:
				break

		logStr = '\r{:03}# {}={:.4f},'.format(epoch, self.lossName, losItem)
		print(logStr, end='')
		tbar.close()
		return losItem#/len(self.dataset.trainSet())

	udh_out = None
	udh_lab = None
	def calc_loss(self, img, lab):
		costList = []
		out = self.model(img)
		# print('run:', img.shape, out[0].shape, lab.shape)
		losSum = self.grad_calc(out, lab, ds=True, criterion=self.criterion)
		costList.append(losSum)
		logStr = 'los={:.4f}'.format(losSum.item())
		if isinstance(out, (list,tuple)):
			out = out[0]
		
		self.udh_out = out
		self.udh_lab = lab
		if self.args.udh:
			losUDH = self.model.regular_udh(out, lab)*self.args.coff_udh
			costList.append(losUDH)
			logStr += ',udh={:.4f}'.format(losUDH.item())
		if self.args.reg:
			losReg = self.model.regular_reg(out, lab)*self.args.coff_reg
			costList.append(losReg)
			logStr += ',reg={:.4f}'.format(losReg.item())
		if self.args.epl:
			losEpl = self.model.regular_epl(out, lab)*self.args.coff_epl
			costList.append(losEpl)
			logStr += ',epl={:.4f}'.format(losEpl.item())
		losSum = sum(costList)
		return losSum, logStr
		