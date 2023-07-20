


def to_onnx(net, in_channels=3, path_onnx='seg_vessel.onnx', device="cpu"):
	dummy_input = torch.randn(1, in_channels, 224, 224, device=device)
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