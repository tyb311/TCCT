# -*- encoding:utf-8 -*-
# 常用资源库
import math,cv2,random
from PIL import Image
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
import onnxruntime as ort




class NetWork:
	def __init__(self, onnx_file=r"tcct_duke.onnx"):
		self.session = ort.InferenceSession(onnx_file)

	tmp={}
	def forward(self, img):
		h,w,c = img.shape
		img = np.transpose(img, (2,0,1))
		img = img.reshape(1,3,h,w).astype(np.float32)/255

		# 预测血管和视杯视盘
		out = self.session.run(None, {"input": img}, )[0].squeeze()
		# out = (out>0.5).astype(np.uint8)*255
		
		print('shape-output:', out.shape)

		return out

		


if __name__ == '__main__':
	img_path = r'oct_duke.png'

	img = np.array(Image.open(img_path).convert('RGB'))[:160, :160]
	print('read:', img.shape)
	
	net = NetWork()
	ret = net.forward(img)

	plt.subplot(121),plt.imshow(img)
	plt.subplot(122),plt.imshow(ret)
	plt.show()