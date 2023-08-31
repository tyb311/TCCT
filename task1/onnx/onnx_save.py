from code import *


def to_onnx(net, in_channels=3, path_onnx='seg_vessel.onnx', device="cpu"):
	dummy_input = torch.randn(1, in_channels, 160, 160, device=device)
	dynamic_axes={'input':{0:'batch',2:'height',3:'width'},'output':{0:'batch',2:'height',3:'width'}}
	# torch.set_grad_enabled(False)
	net.eval()
	print('saving onnx to:', path_onnx)
	with torch.no_grad():
		torch.onnx.export(
			net, dummy_input, path_onnx, verbose=True, 
			input_names=["input"], output_names=["output"],
			dynamic_axes=dynamic_axes, opset_version=11
		)



if __name__ == '__main__':
	device = torch.device('cpu')
	
	# n_class=4
	# print('#'*64, 'Test-Time-Augmentation')
	# model = mpvc().to(device)
	# pth = torch.load(path_pth, map_location=device)
	# msg = model.load_state_dict(pth, strict=False)
	# print('Loaded weight:', path_pth, msg)

	n_class=9
	path_pth = 'tcct_duke.pt'

	# model = mgu(n_class=n_class)
	model = stc_tt(n_class)
	model = RegNet(model, out_channels=n_class).to(device)
	pth = torch.load(path_pth, map_location=device)
	msg = model.load_state_dict(pth, strict=False)
	print('Loaded weight:', path_pth, msg)
	to_onnx(model, path_onnx=path_pth.replace('.pt', '.onnx'))
