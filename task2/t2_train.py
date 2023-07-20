# 导入库
import matplotlib.pyplot as plt
import os, time, random, cv2, skimage
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
# import torchvision.transforms as trans

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet50, resnet34, resnet18

import argparse
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')
parser = argparse.ArgumentParser(description="GOALS Argument")
parser.add_argument('--inc', type=str, default='', help='instruction')
parser.add_argument('--gpu', type=str, default='1', help='cuda number')
parser.add_argument('--net', type=str, default='res18', help='network')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=224, help='image size')
parser.add_argument('--epochs', type=int, default=99, help='epochs')
args = parser.parse_args()


# 设置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
cpu = paddle.CPUPlace()
device = paddle.CPUPlace()
if paddle.device.cuda.device_count()>0:
	paddle.device.set_device('gpu:'+args.gpu)
	device = paddle.CUDAPlace(int(args.gpu))
	# paddle.device.set_device('gpu:0')
print(device)

# 配置
args.root = "./goals_task2"
print(args.root)
os.makedirs(args.root, exist_ok=True)
ckpt_path = f"{args.root}/best_model.pd"


data_root = '/home/tyb/datasets/seteye/goals'
folder_train = f"{data_root}/Train/Image"
folder_infer = f"{data_root}/Validation/Image"

from albumentations import *
import albumentations as alb


# 数据增强
TRANS_PRE = alb.PadIfNeeded(1120,1120)
TRANS_INF = alb.Resize(args.img_size, args.img_size)
TRANS_AUG = alb.Compose([
	alb.Resize(args.img_size,args.img_size),
	alb.RandomResizedCrop(p=1, height=args.img_size, width=args.img_size, scale=(0.6,1.5)),
	alb.HorizontalFlip(p=0.5), alb.VerticalFlip(p=0.5),
	alb.RGBShift(p=1),alb.HueSaturationValue(p=1),alb.RandomContrast(p=1),alb.RandomBrightness(p=1),
	alb.ColorJitter(0.4,0.4,0.4,0.2,p=1),
])

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

		self.file_list = []
		if mode == "infer":
			self.file_list = [[f, 9] for f in os.listdir(dataset_root)]
		else:
			label = {str(row[0]):row[1] for _, row in pd.read_excel(label_file).iterrows()}
			self.file_list = [[f, label[str(int(f.split('.')[0]))]] for f in os.listdir(dataset_root) if f.endswith('.png')]
		
		self.dataset_root = dataset_root
		print(mode, 'data number:', len(self.file_list))
	
	def __getitem__(self, idx):
		idx = idx%100
		real_index, label = self.file_list[idx]
		img_path = os.path.join(self.dataset_root, real_index)  
		# print(img_path)  
		# img_path = img_path.replace('Image', 'Layer_Masks')
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		# print(img.shape, img_path)
		img = TRANS_PRE(image=img)['image']

		if self.mode=='train':
			img = TRANS_AUG(image=img)['image']
		else:
			img = TRANS_INF(image=img)['image']

		# normlize on GPU to save CPU Memory and IO consuming.
		# if not args.mask:
		img = img.astype("float32") / 255.
		img = img.transpose(2, 0, 1) # H, W, C -> C, H, W

		if self.mode == 'infer':
			return img, real_index
		else:            
			return img, label

	def __len__(self):
		if self.mode=='train':
			return len(self.file_list)*20
		return len(self.file_list)


# 网络模型：是否考虑冻结部分参数，如layer2/layer3
class Model(nn.Layer):
	def __init__(self, net='res34', pretrained=True):
		super(Model, self).__init__()
		if 'res34' in net:
			print('#'*32, 'using resnet34')
			base = resnet34(pretrained=pretrained) 
		elif 'res50' in net:
			print('#'*32, 'using resnet50')
			base = resnet50(pretrained=pretrained) 
		else:
			print('#'*32, 'using resnet18')
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

# net = Model()
# x = paddle.rand(shape=(3,3,224,224))
# y = net(x)
# print(y.shape)


# 功能函数
def train(model, train_dataloader, val_dataloader):
	model.train()
	avg_loss_list = []
	avg_acc_list = []
	best_acc = 0.
	# lr = args.root.split('lr')[-1]
	# print('#'*32, 'LR=', lr, int(lr))
	# lr = 10**(-int(lr))
	print('#'*32, 'LR=', args.lr)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = paddle.optimizer.Adam(args.lr, parameters=model.parameters(), weight_decay=5e-4)
	# schedular = lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, cycle_momentum=False, step_size_up=6, step_size_down=24)
	accs_train = []
	for iter in range(args.epochs):
		tbar = tqdm(train_dataloader)
		for data in tbar:
			imgs = data[0]
			
			labs = data[1]#.to(device)
			# print(labs)
			outs = model(imgs)#.to(device)
			pred = F.softmax(outs, axis=1)
			
			loss = criterion(pred, F.one_hot(labs,2))
			avg_loss_list.append(loss.cpu().item())    
			# print(loss.numpy())
			loss.backward()
			optimizer.step()
			optimizer.clear_gradients()

			# print('metric:', pred.shape, labs.shape)
			labs = paddle.unsqueeze(labs, axis=1)
			acc = paddle.metric.accuracy(input=pred, label=labs)
			avg_acc_list.append(acc[0].item())
		tbar.close()
		# schedular.step()

		# if iter % log_interval == 0:
		avg_loss = np.array(avg_loss_list).mean()
		avg_acc_train = np.array(avg_acc_list).mean()
		accs_train.append(avg_acc_train)
		avg_loss_list = []
		avg_acc_list = []
		print("[RUN] iter={}/{} avg_loss={:.4f} avg_acc={:.4f}".format(iter, args.epochs, avg_loss, avg_acc_train))
		# tbar.set_description("[TRAIN] iter={}/{} avg_loss={:.4f} avg_acc={:.4f}".format(iter, args.epochs, avg_loss, avg_acc))

		if iter % 5 == 0:
			scores = valid(model, val_dataloader)
			print("[VAL {}] iter={}/{} auc={:.4f} f1s={:.4f} acc={:.4f}".format(args.root, iter, args.epochs, scores['auc'], scores['f1s'], scores['acc']))
			avg_acc_valid = scores['acc']#
			avg_acc = np.array(accs_train).mean()
			accs_train = []

			if avg_acc_valid>best_acc:
				print('saving better score:{:.4f}->{:.4f}@{:.4f}'.format(avg_acc, best_acc, avg_acc_valid))
				best_acc = avg_acc
				paddle.save(model.state_dict(), ckpt_path)
				
			model.train()

def valid(model, val_dataloader, criterion=None):
	model.eval()
	list_out = []
	list_lab = []
	with paddle.no_grad():
		for data in val_dataloader:
			imgs = data[0]
			labs = data[1]
			# print('valid:', imgs.shape, labs.shape)
			outs = model(imgs)
			pred = F.softmax(outs, axis=1)       
			pred = paddle.argmax(pred, axis=-1).numpy()[0] 
			# print('val:', pred, labs)
			list_out.append(pred)
			list_lab.append(labs.numpy()[0])

			# loss = criterion(pred, F.one_hot(labs,2)).numpy()[0]
			# avg_loss_list.append(loss)

			# labs = paddle.unsqueeze(labs, axis=1)
			# acc = paddle.metric.accuracy(input=pred, label=labs)
			# avg_acc_list.append(acc.numpy())        
	list_out = np.array(list_out)
	list_lab = np.array(list_lab)
	
	auc = metrics.roc_auc_score(list_lab, list_out)
	f1s = metrics.f1_score(list_lab, list_out.round())
	acc = metrics.accuracy_score(list_lab, list_out.round())

	# avg_loss = np.array(avg_loss_list).mean()
	# acc = np.array(avg_acc_list).mean()
	return {'auc':auc, 'f1s':f1s, 'acc':acc}

def predict(model, infer_loader):
	model.eval()

	cache = []
	with paddle.no_grad():
		for i,data in enumerate(infer_loader):
			# if i>99:
			# 	break

			img, idx = data
			# print(idx, img.shape)
			idx = idx[0]
			img = data[0]
			outs = model(img).detach()
			pred = F.softmax(outs, axis=1)
			pred = paddle.argmax(pred, axis=-1).numpy()[0]
			# print('test:', i, pred)
			cache.append([idx, pred])

	csv_pr = f"{args.root}/Classification_Results.csv"
	submission_result = pd.DataFrame(cache, columns=['ImgName', 'GC Pred'])
	submission_result = submission_result.sort_values(by=['ImgName'])
	submission_result[['ImgName', 'GC Pred']].to_csv(csv_pr, index=False)


def setup_seed(seed=311):
	# seed = random.randint(0, 9999)
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	paddle.seed(seed)
	print('> SEEDING DONE')

def main():
	
	train_dataset = GoalClsSet(dataset_root=folder_train, 
							label_file=f'{data_root}/Train/Train_GC_GT.xlsx',
							mode='train'
							)
	valid_dataset = GoalClsSet(dataset_root=folder_train, 
							label_file=f'{data_root}/Train/Train_GC_GT.xlsx',
							mode='valid'
							)
	infer_dataset = GoalClsSet(dataset_root=folder_infer, 
							mode='infer'
							)
					
	train_loader = paddle.io.DataLoader(
		train_dataset,
		num_workers=4,
		batch_size=args.bs, 
		# batch_sampler=paddle.io.DistributedBatchSampler(train_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
		shuffle=False
	)
	valid_loader = paddle.io.DataLoader(
		valid_dataset,
		num_workers=1,
		batch_size=1, 
		# batch_sampler=paddle.io.DistributedBatchSampler(val_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
		shuffle=False
		# return_list=True,
		# use_shared_memory=False
	)
	infer_loader = paddle.io.DataLoader(
		infer_dataset,
		num_workers=1,
		batch_size=1, 
		shuffle=False
	)


	setup_seed()
	model = Model(net=args.net).to(device)

	############################################################
	# 训练阶段
	############################################################
	valid(model, valid_loader)
	train(model, train_loader, valid_loader)


	############################################################
	# 预测阶段
	############################################################
	# model = Model().to(device)
	# paddle.save(model.state_dict(), ckpt_path)
	pt = paddle.load(ckpt_path)
	model.set_state_dict(pt)
	print('submitting begin!')
	print(predict(model, infer_loader))
	print('submitting done!')

	
if __name__ == '__main__':
	main()