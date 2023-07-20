
from loop_seg import KiteSeg
from nets import *
from data import *



import argparse
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser(description="KiteOCT Argument")
parser.add_argument('--db', type=str, default='duke1', choices=['duke','duke1','duke2','duke3','hcms','hcms1','heg','goals', 'odsgh'], help='dataset')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--inc', type=str, default='', help='instruction')
parser.add_argument('--gpu', type=str, default='0', help='cuda number')
parser.add_argument('--los', type=str, default='dice', help='loss function')
parser.add_argument('--net', type=str, default='stc_tt', help='network')
parser.add_argument('--pth', type=str2bool, default=True, help='download pretrained weight!')
# parser.add_argument('--width', type=int, default=32, help='width')
# parser.add_argument('--depth', type=int, default=4, help='depth')

parser.add_argument('--bs', type=int, default=2, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='batch size')
parser.add_argument('--root', type=str, default='', help='folder to train or test again')
parser.add_argument('--resume', type=str2bool, default=False, help='Breakpoint continue to practice!')

parser.add_argument('--reg', type=str2bool, default=False, help='reg loss!')
parser.add_argument('--coff_reg', type=float, default=.1, help='reg loss!')
parser.add_argument('--epl', type=str2bool, default=False, help='Edge Pixel Loss!')
parser.add_argument('--coff_epl', type=float, default=.1, help='Cofficient of MSE!')

parser.add_argument('--udh', type=str2bool, default=False, help='udh loss!')
parser.add_argument('--coff_udh', type=float, default=1, help='udh coff!')
parser.add_argument('--type_udh', type=str, default='cos', choices=['cos','mse'], help='udh loss!')

parser.add_argument('--ds', type=str2bool, default=False, help='Deep Supervision!')
parser.add_argument('--coff_ds', type=float, default=1, help='deep supervision!')

parser.add_argument('--pl', type=str2bool, default=False, help='Parrallel!')
parser.add_argument('--bug', type=str2bool, default=False, help='Debug Mode!')
args = parser.parse_args()



if __name__ == '__main__':
	ALB_TWIST = make_tran(256,256)
	dataset = EyeSetGenerator(dbname=args.db)
	
	# 模型
	##################################################################
	net = eval(args.net+'(dataset.out_channels)')
	net = RegNet(net, con=args.type_udh, out_channels=dataset.out_channels)
	print('OUT-CHANNELS:', dataset.out_channels)


	#	实验
	##################################################################
	keras = KiteSeg(model=net, dataset=dataset, root=args.root, args=args) 
	if args.resume:
		path = args.root+'/val_iou.pt'
		pt = torch.load(path, map_location=keras.device)
		keras.model.load_state_dict(pt, strict=False)
		print('loaded model:', path)
	keras.fit(epochs=1 if args.bug else args.epochs)
