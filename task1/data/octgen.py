# -*- coding: utf-8 -*-

from .tran import *
from .utils import *
from .octnpy import *

# SIZE_IMAGE=384
SIZE_IMAGEH, SIZE_IMAGEW = 256,256#224,224#256, 448#128,128#
def make_tran(SIZE_IMAGEH,SIZE_IMAGEW):
	ALB_TWIST = alb.Compose([
		alb.PadIfNeeded(SIZE_IMAGEH, SIZE_IMAGEW,p=1, border_mode=cv2.BORDER_CONSTANT, mask_value=0),
		alb.CropNonEmptyMaskIfExists(SIZE_IMAGEH,SIZE_IMAGEW,p=1),

		alb.HorizontalFlip(p=0.5), 
		alb.VerticalFlip(p=0.5),
		alb.RGBShift(p=1),alb.HueSaturationValue(p=1),alb.RandomContrast(p=1),alb.RandomBrightness(p=1),
	])
	return ALB_TWIST
ALB_TWIST = make_tran(SIZE_IMAGEH,SIZE_IMAGEW)

ALB_VALID = alb.Compose([
	alb.HorizontalFlip(p=1), 
	alb.VerticalFlip(p=0.5),
	# alb.RGBShift(p=1),alb.HueSaturationValue(p=1),alb.RandomContrast(p=1),alb.RandomBrightness(p=1),
])

from torch.utils.data import DataLoader, Dataset
class EyeSetGenerator(Dataset, EyeSetResource):
	exeNums = {'train':8, 'val':1, 'test':1}#Aug4Val.number
	exeMode = 'train'#train, val, test
	exeData = 'train'#train, test, full
	in_channels = 3
	out_channels= 8
	
	def __init__(self, **args):
		super(EyeSetGenerator, self).__init__(**args)

		if self.__name__ == 'hcms':
			self.exeNums['train'] = 1
			self.out_channels = 9
		elif self.__name__ == 'hcms1':
			self.exeNums['train'] = 1
			self.out_channels = 9
		elif self.__name__ == 'duke':
			self.exeNums['train'] = 8#13.3610
			self.out_channels = 9
		elif self.__name__ == 'duke1':
			self.exeNums['train'] = 8#13.3610
			self.out_channels = 9
		elif self.__name__ == 'duke2':
			self.exeNums['train'] = 8#13.3610
			self.out_channels = 9
		elif self.__name__ == 'duke3':
			self.exeNums['train'] = 14#13.3610
			self.out_channels = 9
			
		elif self.__name__ == 'heg':
			self.exeNums['train'] = 15#14.7#12
			self.out_channels = 8
		elif self.__name__ == 'goals':
			self.exeNums['train'] = 15#14.7#12
			self.out_channels = 5
		
		self.exeNums['train'] = max(1,735 // self.lens['train'])
		print('exeNums:', self.exeNums)

	def __len__(self):
		if self.isTestMode:
			return self.lens['test']#*self.exeNums[self.exeMode]
		elif self.isValMode:
			return self.lens['val']#*50
			# return 1024//self.lens['val']*self.lens['val']
		return self.lens['train']*self.exeNums[self.exeMode]

	def set_mode(self, mode='train'):
		self.exeMode = mode
		self.exeData = 'train' if mode!='test' else 'test'
		self.isTrainMode = (mode=='train')
		self.isValMode = (mode=='val')
		self.isTestMode = (mode=='test')
	def trainSet(self, bs=32, data='train'):#pin_memory=True, , shuffle=True
		self.set_mode(mode='train')
		return DataLoader(self, batch_size=bs, pin_memory=True, num_workers=4, shuffle=True)
	def valSet(self, bs=1, data='val'):
		self.set_mode(mode='val')
		return DataLoader(self, batch_size=bs,  pin_memory=True, num_workers=1)
	def testSet(self, bs=1, data='test'):
		self.set_mode(mode='test')
		return DataLoader(self, batch_size=bs,  pin_memory=True, num_workers=1)
	#DataLoader worker (pid(s) 5220) exited unexpectedly, 令numworkers>1就可以啦
	
	def parse(self, pics):
		return pics['img'], pics['lab'], pics['tag'], None

	def __getitem__(self, idx):
		# print(self.exeData, self.exeMode)
		# idx = idx % self.lens[self.exeData] 

		# flg = idx //self.lens['train']
		if self.isTestMode:
			idx = idx %self.lens['test']
			path_img = self.inf_oct[idx]
			path_out = self.inf_lab[idx]
		elif self.isValMode:
			idx = idx %self.lens['val']
			path_img = self.val_oct[idx]
			path_out = self.val_lab[idx]
		else:
			idx = idx %self.lens['train']
			path_img = self.src_oct[idx]
			path_out = self.src_lab[idx]
		# print('images:', path_img, path_out)
		pics = self.readPair(path_img, path_out)

		if self.isTrainMode:

			pair = ALB_TWIST(image=pics['img'], mask=pics['lab'])
			pics['img'],pics['lab'] = pair['image'], pair['mask']
			
		elif self.isValMode:
			pair = ALB_VALID(image=pics['img'], mask=pics['lab'])
			pics['img'],pics['lab'] = pair['image'], pair['mask']

		img = torch.from_numpy(pics['img']).permute(2,0,1).float()/255
		lab = torch.from_numpy(pics['lab']).long()
		pics = {'img':img.clamp(0,1), 'lab':lab, 'tag':path_out}
		# print('__getitem__:', pics['img'].shape, pics['lab'].shape)
		return pics
		