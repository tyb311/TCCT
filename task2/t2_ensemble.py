# 导入库
import matplotlib.pyplot as plt
import os, time, random, cv2, skimage
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from glob import glob
import warnings
warnings.filterwarnings('ignore')

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet50, resnet34, resnet18

import albumentations as alb

# 数据增强
TRANS_PRE = alb.PadIfNeeded(1120,1120)
TRANS_INF = alb.Resize(224,224)
# 数据加载
class GoalClsSet(paddle.io.Dataset):
	def __init__(self,
				dataset_root,
				label_file='',
				mode='train',
				**args):
		super(GoalClsSet, self).__init__(**args)
		self.dataset_root = dataset_root
		self.mode = mode.lower()

		if mode == "infer":
			self.file_list = [[f, 9] for f in os.listdir(dataset_root)]
		else:
			if label_file.endswith('.csv'):
				df = pd.read_csv(label_file)
			else:
				df = pd.read_excel(label_file)
			label = {str(row[0]):row[1] for _, row in df.iterrows()}
			if mode == "infer":
				self.file_list = [[f, label['{:04d}.png'.format(int(f.split('.')[0]))]] for f in os.listdir(dataset_root) if f.endswith('.png')]
			else:
				self.file_list = [[f, label[str(int(f.split('.')[0]))]] for f in os.listdir(dataset_root) if f.endswith('.png')]
		
		self.dataset_root = dataset_root
		print(mode, 'set:', len(self.file_list))
	
	trans_val = [
		alb.Compose([alb.HorizontalFlip(p=1),alb.HorizontalFlip(p=1)]),
		alb.Compose([alb.HorizontalFlip(p=1),alb.VerticalFlip(p=1)]),
		alb.HorizontalFlip(p=1),alb.VerticalFlip(p=1),
	]
	def __getitem__(self, idx):
		filename, label = self.file_list[idx%100]
		img_path = os.path.join(self.dataset_root, filename)  
		# print(img_path)  
		# img_path = img_path.replace('Image', 'Layer_Masks')
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		# print(img.shape, img_path)
		img = TRANS_PRE(image=img)['image']

		if self.mode=='train':
			img = TRANS_AUG(image=img)['image']
		else:
			img = TRANS_INF(image=img)['image']
		
		# if self.mode=='valid':
			flag_tran = idx // self.file_list.__len__()
			img = self.trans_val[flag_tran%4](image=img)['image']

		# normlize on GPU to save CPU Memory and IO consuming.
		# if not args.mask:
		img = img.astype("float32") / 255.
		img = img.transpose(2, 0, 1) # H, W, C -> C, H, W

		return img, label, filename

	def __len__(self):
		if self.mode=='train':
			return len(self.file_list)*20
		return len(self.file_list)


# 网络模型：是否考虑冻结部分参数，如layer2/layer3
class Model(nn.Layer):
	def __init__(self, net='res34', pretrained=True):
		super(Model, self).__init__()
		if 'res34' in net:
			# print('#'*32, 'using resnet34')
			base = resnet34(pretrained=pretrained) 
		elif 'res50' in net:
			# print('#'*32, 'using resnet50')
			base = resnet50(pretrained=pretrained) 
		else:
			# print('#'*32, 'using resnet18')
			base = resnet18(pretrained=pretrained) 
		# print(base)
		base.layer1 = nn.Sequential(base.layer1, nn.Dropout(0.1))
		base.layer2 = nn.Sequential(base.layer2, nn.Dropout(0.2))
		base.layer3 = nn.Sequential(base.layer3, nn.Dropout(0.3))
		base.layer4 = nn.Sequential(base.layer4, nn.Dropout(0.4))
		# print(base.fc.weight.shape)
		base.fc = nn.Sequential(
			nn.Linear(base.fc.weight.shape[0],256),
			nn.Linear(256,2)
		)
		self.base = base

	def forward(self, img):
		return self.base(img)

def predict(model, img):
	outs = model(img).detach()
	pred = F.softmax(outs, axis=1)
	pred = paddle.argmax(pred, axis=-1).numpy()[0]
	return pred

def test(model, data_loader, i):
	model.eval()

	cache = []
	with paddle.no_grad():
		tbar = tqdm(data_loader)
		for k,data in enumerate(tbar):
			img, _, file = data
			# print(file, img.shape)
			idx = file[0]
			img = data[0]
			pred0 = predict(model, img)
			pred1 = predict(model, paddle.flip(img, axis=2))
			pred2 = predict(model, paddle.flip(img, axis=3))
			pred3 = predict(model, paddle.flip(img, axis=(2,3)))
			pred = (pred0+pred1+pred2+pred3)/4
			# print('test:', i, pred)
			cache.append([idx, pred])
			tbar.set_description('tta4-index:{}-{}-{}'.format(i, pred, round(pred)))
		tbar.close()
	csv_pr = f"task2/prediction/Classification_Results{i}.csv"
	submission_result = pd.DataFrame(cache, columns=['ImgName', 'GC Pred'])
	submission_result = submission_result.sort_values(by=['ImgName'])
	submission_result[['ImgName', 'GC Pred']].to_csv(csv_pr, index=False)
	# print(i, 'save to:', csv_pr)



data_root = '/home/tyb/datasets/seteye/goals'
data_root = r'G:\Objects\Cometition\dataset\GOALS2022\goals'
folder_train = f"{data_root}/Train/Image"
folder_infer = f"{data_root}/Validation/Image"
infer_dataset = GoalClsSet(dataset_root=folder_infer, mode='infer')
infer_loader = paddle.io.DataLoader(
	infer_dataset,
	num_workers=1,
	batch_size=1, 
	shuffle=False
)
if __name__ == '__main__':
	#	prediction
	#########################################################################
	for i,path_ckpt in enumerate(glob(r'task2\weights\*.pd')):

		if 'res18' in path_ckpt:
			tag = 'res18'
		elif 'res34' in path_ckpt:
			tag = 'res34'
		else:
			tag = 'res50'

		model = Model(net=tag)
		pt = paddle.load(path_ckpt)
		model.set_state_dict(pt)

		test(model, infer_loader, i)
		


	#	ensemble
	#########################################################################
	cls_pr = np.zeros(shape=(100,))
	csvs_pr = glob(r'task2\prediction\*.csv')
	for csv_pr in csvs_pr:
		df_pr = pd.read_csv(csv_pr)
		df_pr = df_pr.sort_values(by=['ImgName'])
		# print(df_pr.head())
		cls_pr += df_pr['GC Pred']
	cls_pr = (cls_pr / len(csvs_pr)).round()
	df_pr = pd.DataFrame.from_dict({'ImgName':df_pr['ImgName'], 'GC Pred':cls_pr})
	df_pr.to_csv('task2/Classification_Results.csv', index=False)
	print(df_pr.head())
