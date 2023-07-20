# coding: utf-8
import numpy as np
EPS = 1e-9#np.spacing(1)
import os,glob,math,cv2,random,numbers
from torchvision import transforms
from torchvision.transforms import functional as f
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .tran import *


'''
duke	536*496		高度不定
hcms	1024*496
heg		610*496
'''

import albumentations as alb
from albumentations import PadIfNeeded, CenterCrop, Resize
class EyeSetResource(object):
	size = dict()
	def __init__(self, dbname='goals', **args):
		super(EyeSetResource, self).__init__()
		print('\n', '#'*32)
		
		self.__name__ = dbname

		self.folder = r'G:\Objects\Cometition\dataset\OCTSets'

		self.folder += '/'+dbname
		print('Folder of Dataset:', self.folder)
		
		self.src_oct = list(sorted([it.replace('\\','/') for it in glob.glob(self.folder+'/train_img/*/*.*')]))\
			+ list(sorted([it.replace('\\','/') for it in glob.glob(self.folder+'/train_img/*.*')]))
		self.val_oct = list(sorted([it.replace('\\','/') for it in glob.glob(self.folder+'/val_img/*/*.*')]))\
			+ list(sorted([it.replace('\\','/') for it in glob.glob(self.folder+'/val_img/*.*')]))
		if self.val_oct.__len__()==0:
			self.val_oct = self.src_oct.copy()
		self.inf_oct = list(sorted([it.replace('\\','/') for it in glob.glob(self.folder+'/test_img/*/*.*')]))\
			+ list(sorted([it.replace('\\','/') for it in glob.glob(self.folder+'/test_img/*.*')]))

		self.src_lab = [oct.replace('train_img', 'train_lab') for oct in self.src_oct]
		self.val_lab = [oct.replace('val_img', 'val_lab').replace('train_img', 'train_lab') for oct in self.val_oct]
		self.inf_lab = [oct.replace('test_img', 'test_lab') for oct in self.inf_oct]

		self.lens = {
			'train':len(self.src_lab),   
			'val':len(self.val_lab),  
			'test':len(self.inf_oct),
			}  
		print(self.__name__, self.lens)
		# print(f'Number of {self.__name__}:', self.lens)  
		# print('*'*32,'eyeset','*'*32)

		if dbname=='heg':
			self.height_stt, self.height_end = 83, 339
			self.prep_tran = alb.PadIfNeeded(p=1, min_height=256, min_width=672, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0)
		elif dbname=='duke':
			self.height_stt, self.height_end = 0, 224
			self.prep_tran = alb.PadIfNeeded(p=1, min_height=256, min_width=576, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0)
		elif dbname=='duke1' or dbname=='duke3':
			self.height_stt, self.height_end = 0, 224
			self.prep_tran = alb.PadIfNeeded(p=1, min_height=256, min_width=576, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0)
		elif dbname=='duke2':	
			self.height_stt, self.height_end = 0, 384
			self.prep_tran = alb.PadIfNeeded(p=1, min_height=384, min_width=576, border_mode=cv2.BORDER_REFLECT, mask_value=0, value=0)
		elif dbname=='hcms':
			self.height_stt, self.height_end = 0, 1024
			self.prep_tran = alb.Resize(p=1, height=256, width=512, interpolation=cv2.INTER_NEAREST)
			self.post_tran = alb.Resize(p=1, height=128, width=1024, interpolation=cv2.INTER_NEAREST)
			# self.prep_tran = alb.Resize(p=1, height=128, width=1024, interpolation=cv2.INTER_NEAREST)
			# self.post_tran = alb.Resize(p=1, height=128, width=1024, interpolation=cv2.INTER_NEAREST)
		elif dbname=='hcms1':
			self.height_stt, self.height_end = 0, 1024
			self.prep_tran = alb.Resize(p=1, height=256, width=512, interpolation=cv2.INTER_NEAREST)
			self.post_tran = alb.Resize(p=1, height=128, width=1024, interpolation=cv2.INTER_NEAREST)
			# self.prep_tran = alb.PadIfNeeded(p=1, min_height=256, min_width=1024, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0)
			# self.post_tran = alb.CenterCrop(p=1, height=128, width=1024)
		elif dbname=='goals':
			self.height_stt, self.height_end = 0, 608
			self.prep_tran = alb.Resize(p=1, height=608, width=512, interpolation=cv2.INTER_NEAREST)
			self.post_tran = alb.Resize(p=1, height=608, width=1100, interpolation=cv2.INTER_NEAREST)
		else:#odsgh
			self.height_stt, self.height_end = 0, 992
			self.prep_tran = alb.Resize(p=1, height=496, width=512, interpolation=cv2.INTER_NEAREST)
			self.post_tran = alb.Resize(p=1, height=992, width=1024, interpolation=cv2.INTER_NEAREST)

	post_tran = alb.Compose([
		alb.Resize(height=672, width=1152, interpolation=cv2.INTER_NEAREST),
		alb.CenterCrop(height=672, width=1100),
	])
	def postprocess(self, img, tag, return_lab=False):
		img = (img*self.divide).astype(np.uint8)
		# print(tag)
		lab = cv2.imread(tag, cv2.IMREAD_GRAYSCALE)
		h,w = lab.shape[-2:]
		bgd = np.zeros_like(lab)
		if self.__name__ in['heg','duke','duke1','duke2','duke3']:
			h = min(min(h,lab.shape[0]),img.shape[0])
			w = min(min(w,lab.shape[1]),img.shape[1])
			post_tran = alb.CenterCrop(height=h, width=w)
		else:
			post_tran = self.post_tran
		img = post_tran(image=img)['image']
		# print(img.shape, lab.shape)
		bgd[self.height_stt:self.height_end,:] = img
		if return_lab:
			return bgd, lab
		return bgd

	height_stt=0
	height_end = 9999
	divide = 30
	def readPair(self, img, lab):
		# print('readPair:', img, lab)  
		img = cv2.imread(img, cv2.IMREAD_COLOR)
		# print('read-0:', img.shape)
		img = img[self.height_stt:self.height_end,:]
		# print('read-1:', img.shape)
		lab = cv2.imread(lab, cv2.IMREAD_GRAYSCALE)//self.divide
		lab = lab[self.height_stt:self.height_end,:]
		# print('readPair:', img.shape, lab.shape)
		img = self.prep_tran(image=img)['image']
		# print('read-2:', img.shape)
		lab = self.prep_tran(image=lab)['image']
		return {'img':img, 'lab':lab,}
		
		