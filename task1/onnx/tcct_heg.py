# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4)
import os,glob,numbers,math,cv2,random,socket,shutil
EPS = 1e-6#np.spacing(1)#

# 图像处理
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn.functional as F
def norm_add(xs, size=None):
	# for x in xs:
	# 	print(x.shape)
	xs = [F.normalize(x, dim=1, p=2) for x in xs]
	xs = [F.interpolate(x, size=xs[0].shape[-2:], mode='bilinear', align_corners=False) for x in xs]
	return [sum(xs)/len(xs)]

class BasicBlock(nn.Module):
	'''
	param ={
		'num_channels':1,
		'num_filters':64,
		'kernel_h':7,
		'kernel_w':3,
		'stride_conv':1,
		'pool':2,
		'stride_pool':2,
		'num_classes':10
	}
	'''
	def __init__(self, params):
		super(BasicBlock, self).__init__()

		padding_h = int((params['kernel_h'] - 1) / 2)
		padding_w = int((params['kernel_w'] - 1) / 2)

		self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
							  kernel_size=(params['kernel_h'], params['kernel_w']),
							  padding=(padding_h, padding_w),
							  stride=params['stride_conv'])
		self.batchnorm = nn.BatchNorm2d(num_features=params['num_filters'])
		self.prelu = nn.PReLU()

	def forward(self, input):
		out_conv = self.conv(input)
		out_bn = self.batchnorm(out_conv)
		out_prelu = self.prelu(out_bn)
		return out_prelu

class EncoderBlock(BasicBlock):
	def __init__(self, params):
		super(EncoderBlock, self).__init__(params)
		self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

	def forward(self, input):
		out_block = super(EncoderBlock, self).forward(input)
		out_encoder, indices = self.maxpool(out_block)
		return out_encoder, out_block, indices

class DecoderBlock(BasicBlock):
	def __init__(self, params):
		super(DecoderBlock, self).__init__(params)
		self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

	def forward(self, input, out_block, indices):
		unpool = self.unpool(input, indices)
		concat = torch.cat((out_block, unpool), dim=1)
		out_block = super(DecoderBlock, self).forward(concat)

		return out_block

class ClassifierBlock(nn.Module):
	def __init__(self, params):
		super(ClassifierBlock, self).__init__()
		self.conv = nn.Conv2d(params['num_channels'], params['num_class'], 1, 1)
		# self.softmax = nn.Softmax2d()

	def forward(self, input):
		out_conv = self.conv(input)
		#out_logit = self.softmax(out_conv)
		return out_conv

class ReLayNet(nn.Module):
	__name__='relay'
	def __init__(self, in_channels=3, out_channels=8):
		super(ReLayNet, self).__init__()

		params ={
			'num_filters':64,
			'num_channels':in_channels,
			'kernel_h':7,
			'kernel_w':3,
			'kernel_c':3,
			'stride_conv':1,
			'pool':2,
			'stride_pool':2,
			'num_class':out_channels
		}
		self.encode1 = EncoderBlock(params)
		params['num_channels'] = 64
		self.encode2 = EncoderBlock(params)
		# params['num_channels'] = 64  # This can be used to change the numchannels for each block
		self.encode3 = EncoderBlock(params)
		self.bottleneck = BasicBlock(params)
		params['num_channels'] = 128
		self.decode1 = DecoderBlock(params)
		self.decode2 = DecoderBlock(params)
		self.decode3 = DecoderBlock(params)
		params['num_channels'] = 64
		self.classifier = ClassifierBlock(params)

		self.eran1 = nn.Conv2d(64,32,1,1)
		self.eran2 = nn.Conv2d(64,32,1,1)
		self.eran3 = nn.Conv2d(64,32,1,1)
		self.dran1 = nn.Conv2d(64,32,1,1)
		self.dran2 = nn.Conv2d(64,32,1,1)
		self.dran3 = nn.Conv2d(64,32,1,1)

	feats=None
	def forward(self, input):
		e1, out1, ind1 = self.encode1.forward(input)
		e2, out2, ind2 = self.encode2.forward(e1)
		e3, out3, ind3 = self.encode3.forward(e2)
		bn = self.bottleneck.forward(e3)

		d3 = self.decode1.forward(bn, out3, ind3)
		d2 = self.decode2.forward(d3, out2, ind2)
		d1 = self.decode3.forward(d2, out1, ind1)
		prob = self.classifier.forward(d1)

		e1 = F.interpolate(e1, size=input.shape[-2:], mode='bilinear')
		self.feats = norm_add([
			self.eran1(e1),self.eran2(e2),self.eran3(e3),
			self.dran1(d1),self.dran2(d2),self.dran3(d3)])

		return prob

def relay(n_class=8):
	return ReLayNet(3, n_class)

from functools import partial
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from torch import einsum, nn

DROP_RATE=0.0

class Mlp(nn.Module):
	def __init__(
		self,
		in_features,
		hidden_features=None,
		out_features=None,
		act_layer=nn.GELU,
		drop=DROP_RATE,
	):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		"""foward function"""
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x

class Conv2d_BN(nn.Module):
	"""Convolution with BN module."""
	def __init__(
		self,
		in_ch,
		out_ch,
		kernel_size=1,
		stride=1,
		pad=0,
		dilation=1,
		groups=1,
		bn_weight_init=1,
		norm_layer=nn.BatchNorm2d,
		act_layer=None,
	):
		super().__init__()

		self.conv = torch.nn.Conv2d(in_ch,
									out_ch,
									kernel_size,
									stride,
									pad,
									dilation,
									groups,
									bias=False)
		self.bn = norm_layer(out_ch)
		torch.nn.init.constant_(self.bn.weight, bn_weight_init)
		torch.nn.init.constant_(self.bn.bias, 0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# Note that there is no bias due to BN
				fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

		self.act_layer = act_layer() if act_layer is not None else nn.Identity(
		)

	def forward(self, x):
		"""foward function"""
		x = self.conv(x)
		x = self.bn(x)
		x = self.act_layer(x)
		return x

class DWConv2d_BN(nn.Module):
	"""Depthwise Separable Convolution with BN module."""
	def __init__(
		self,
		in_ch,
		out_ch,
		kernel_size=1,
		stride=1,
		norm_layer=nn.BatchNorm2d,
		act_layer=nn.Hardswish,
		bn_weight_init=1,
	):
		super().__init__()

		# dw
		self.dwconv = nn.Conv2d(
			in_ch,
			out_ch,
			kernel_size,
			stride,
			(kernel_size - 1) // 2,
			groups=out_ch,
			bias=False,
		)
		# pw-linear
		self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
		self.bn = norm_layer(out_ch)
		self.act = act_layer() if act_layer is not None else nn.Identity()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2.0 / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(bn_weight_init)
				m.bias.data.zero_()

	def forward(self, x):
		"""
		foward function
		"""
		x = self.dwconv(x)
		x = self.pwconv(x)
		x = self.bn(x)
		x = self.act(x)

		return x

class DWCPatchEmbed(nn.Module):
	"""Depthwise Convolutional Patch Embedding layer Image to Patch
	Embedding."""
	def __init__(self,
				 in_chans=3,
				 embed_dim=768,
				 patch_size=16,
				 stride=1,
				 act_layer=nn.Hardswish):
		super().__init__()

		self.patch_conv = DWConv2d_BN(
			in_chans,
			embed_dim,
			kernel_size=patch_size,
			stride=stride,
			act_layer=act_layer,
		)

	def forward(self, x):
		"""foward function"""
		x = self.patch_conv(x)
		return x

class Patch_Embed_stage(nn.Module):
	"""Depthwise Convolutional Patch Embedding stage comprised of
	`DWCPatchEmbed` layers."""
	def __init__(self, embed_dim, num_path=4, isPool=False):
		super(Patch_Embed_stage, self).__init__()

		self.patch_embeds = nn.ModuleList([
			DWCPatchEmbed(
				in_chans=embed_dim,
				embed_dim=embed_dim,
				patch_size=3,
				stride=2 if isPool and idx == 0 else 1,
			) for idx in range(num_path)
		])

	def forward(self, x):
		"""foward function"""
		att_inputs = []
		for pe in self.patch_embeds:
			x = pe(x)
			att_inputs.append(x)

		return att_inputs

class ConvPosEnc(nn.Module):
	"""Convolutional Position Encoding.

	Note: This module is similar to the conditional position encoding in CPVT.
	"""
	def __init__(self, dim, k=3):
		"""init function"""
		super(ConvPosEnc, self).__init__()

		self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

	def forward(self, x, size):
		"""foward function"""
		B, N, C = x.shape
		H, W = size

		feat = x.transpose(1, 2).view(B, C, H, W)
		x = self.proj(feat) + feat
		x = x.flatten(2).transpose(1, 2)

		return x

class ConvRelPosEnc(nn.Module):
	"""Convolutional relative position encoding."""
	def __init__(self, Ch, h, window):
		"""Initialization.

		Ch: Channels per head.
		h: Number of heads.
		window: Window size(s) in convolutional relative positional encoding.
				It can have two forms:
				1. An integer of window size, which assigns all attention heads
				   with the same window size in ConvRelPosEnc.
				2. A dict mapping window size to #attention head splits
				   (e.g. {window size 1: #attention head split 1, window size
									  2: #attention head split 2})
				   It will apply different window size to
				   the attention head splits.
		"""
		super().__init__()

		if isinstance(window, int):
			# Set the same window size for all attention heads.
			window = {window: h}
			self.window = window
		elif isinstance(window, dict):
			self.window = window
		else:
			raise ValueError()

		self.conv_list = nn.ModuleList()
		self.head_splits = []
		for cur_window, cur_head_split in window.items():
			dilation = 1  # Use dilation=1 at default.
			padding_size = (cur_window + (cur_window - 1) *
							(dilation - 1)) // 2
			cur_conv = nn.Conv2d(
				cur_head_split * Ch,
				cur_head_split * Ch,
				kernel_size=(cur_window, cur_window),
				padding=(padding_size, padding_size),
				dilation=(dilation, dilation),
				groups=cur_head_split * Ch,
			)
			self.conv_list.append(cur_conv)
			self.head_splits.append(cur_head_split)
		self.channel_splits = [x * Ch for x in self.head_splits]

	def forward(self, q, v, size):
		"""foward function"""
		B, h, N, Ch = q.shape
		H, W = size

		# We don't use CLS_TOKEN
		q_img = q
		v_img = v

		# Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
		v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
		# Split according to channels.
		v_img_list = torch.split(v_img, self.channel_splits, dim=1)
		conv_v_img_list = [
			conv(x) for conv, x in zip(self.conv_list, v_img_list)
		]
		conv_v_img = torch.cat(conv_v_img_list, dim=1)
		# Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
		conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

		EV_hat_img = q_img * conv_v_img
		EV_hat = EV_hat_img
		return EV_hat

class FactorAtt_ConvRelPosEnc(nn.Module):
	"""Factorized attention with convolutional relative position encoding
	class."""
	def __init__(
		self,
		dim,
		num_heads=8,
		qkv_bias=False,
		qk_scale=None,
		attn_drop=DROP_RATE,
		proj_drop=DROP_RATE,
		shared_crpe=None,
	):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim**-0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		# Shared convolutional relative position encoding.
		self.crpe = shared_crpe

	def forward(self, x, size):
		"""foward function"""
		B, N, C = x.shape

		# Generate Q, K, V.
		qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
								   C // self.num_heads).permute(2, 0, 3, 1, 4))
		q, k, v = qkv[0], qkv[1], qkv[2]

		# Factorized attention.
		k_softmax = k.softmax(dim=2)
		k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
		factor_att = einsum("b h n k, b h k v -> b h n v", q,
							k_softmax_T_dot_v)

		# Convolutional relative position encoding.
		crpe = self.crpe(q, v, size=size)

		# Merge and reshape.
		x = self.scale * factor_att + crpe
		x = x.transpose(1, 2).reshape(B, N, C)

		# Output projection.
		x = self.proj(x)
		x = self.proj_drop(x)

		return x

class HydraAttention(nn.Module):
	"""Factorized attention with convolutional relative position encoding
	class."""
	def __init__(
		self,
		dim,
		num_heads=8,
		qkv_bias=False,
		qk_scale=None,
		attn_drop=DROP_RATE,
		proj_drop=DROP_RATE,
		shared_crpe=None,
	):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim**-0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		# Shared convolutional relative position encoding.
		self.crpe = shared_crpe

	def hydra(self, q, k, v):
		"""
		q, k, and v should all be tensors of shape
		[batch, tokens, features]
		"""
		q = q / q.norm(dim=-1, keepdim=True)
		k = k / k.norm(dim=-1, keepdim=True)
		kv = (k * v).sum(dim=-2, keepdim=True)
		out = q * kv
		return out

	def forward(self, x, size):
		"""foward function"""
		B, N, C = x.shape

		# Generate Q, K, V.
		qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
								   C // self.num_heads).permute(2, 0, 3, 1, 4))
		q, k, v = qkv[0], qkv[1], qkv[2]

		# Hydra attention.
		factor_att = self.hydra(q,k,v)

		# Convolutional relative position encoding.
		crpe = self.crpe(q, v, size=size)

		# Merge and reshape.
		x = self.scale * factor_att + crpe
		x = x.transpose(1, 2).reshape(B, N, C)

		# Output projection.
		x = self.proj(x)
		x = self.proj_drop(x)

		return x

class MetaPool(nn.Module):
	"""
	Implementation of pooling for PoolFormer
	--pool_size: pooling size
	"""
	def __init__(self, pool_size=3, **kwargs):
		super().__init__()
		self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

	def forward(self, x, size=None):
		return self.pool(x) - x

class MHCABlock(nn.Module):
	"""Multi-Head Convolutional self-Attention block."""
	def __init__(
		self,
		dim,
		num_heads,
		mlp_ratio=3,
		drop_path=0.0,
		qkv_bias=True,
		qk_scale=None,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		shared_cpe=None,
		shared_crpe=None,
	):
		super().__init__()

		self.cpe = shared_cpe
		self.crpe = shared_crpe
		# self.att = HydraAttention(
		# 	dim,
		# 	num_heads=num_heads,
		# 	qkv_bias=qkv_bias,
		# 	qk_scale=qk_scale,
		# 	shared_crpe=shared_crpe,
		# )
		# self.att = FactorAtt_ConvRelPosEnc(
		# 	dim,
		# 	num_heads=num_heads,
		# 	qkv_bias=qkv_bias,
		# 	qk_scale=qk_scale,
		# 	shared_crpe=shared_crpe,
		# )
		self.att = MetaPool()

		self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
		self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

		self.norm1 = norm_layer(dim)
		self.norm2 = norm_layer(dim)

	def forward(self, x, size):
		"""foward function"""
		if self.cpe is not None:
			x = self.cpe(x, size)
		cur = self.norm1(x)
		# print('ATT-before:', cur.shape)
		a = self.att(cur, size)
		# print('ATT-after:', a.shape)
		x = x + self.drop_path(a)

		cur = self.norm2(x)
		x = x + self.drop_path(self.mlp(cur))
		return x

class MHCAEncoder(nn.Module):
	"""Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
	blocks."""
	def __init__(
		self,
		dim,
		num_layers=1,
		num_heads=8,
		mlp_ratio=3,
		drop_path_list=[],
		qk_scale=None,
		crpe_window={
			3: 2,
			5: 3,
			7: 3
		},
	):
		super().__init__()

		self.num_layers = num_layers
		self.cpe = ConvPosEnc(dim, k=3)
		self.crpe = ConvRelPosEnc(Ch=dim // num_heads,
								  h=num_heads,
								  window=crpe_window)
		self.MHCA_layers = nn.ModuleList([
			MHCABlock(
				dim,
				num_heads=num_heads,
				mlp_ratio=mlp_ratio,
				drop_path=drop_path_list[idx],
				qk_scale=qk_scale,
				shared_cpe=self.cpe,
				shared_crpe=self.crpe,
			) for idx in range(self.num_layers)
		])

	def forward(self, x, size):
		"""foward function"""
		H, W = size
		B = x.shape[0]
		for layer in self.MHCA_layers:
			x = layer(x, (H, W))

		# return x's shape : [B, N, C] -> [B, C, H, W]
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		return x

class ResBlock(nn.Module):
	"""Residual block for convolutional local feature."""
	def __init__(
		self,
		in_features,
		hidden_features=None,
		out_features=None,
		act_layer=nn.Hardswish,
		norm_layer=nn.BatchNorm2d,
	):
		super().__init__()

		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.conv1 = Conv2d_BN(in_features,
							   hidden_features,
							   act_layer=act_layer)
		self.dwconv = nn.Conv2d(
			hidden_features,
			hidden_features,
			3,
			1,
			1,
			bias=False,
			groups=hidden_features,
		)
		self.norm = norm_layer(hidden_features)
		self.act = act_layer()
		self.conv2 = Conv2d_BN(hidden_features, out_features)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		"""
		initialization
		"""
		if isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()

	def forward(self, x):
		"""foward function"""
		identity = x
		feat = self.conv1(x)
		feat = self.dwconv(feat)
		feat = self.norm(feat)
		feat = self.act(feat)
		feat = self.conv2(feat)

		return identity + feat

class MHCA_stage(nn.Module):
	"""Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
	layers."""
	def __init__(
		self,
		embed_dim,
		out_embed_dim,
		num_layers=1,
		num_heads=8,
		mlp_ratio=3,
		num_path=4,
		drop_path_list=[],
	):
		super().__init__()

		self.mhca_blks = nn.ModuleList([
			MHCAEncoder(
				embed_dim,
				num_layers,
				num_heads,
				mlp_ratio,
				drop_path_list=drop_path_list,
			) for _ in range(num_path)
		])

		self.InvRes = ResBlock(in_features=embed_dim, out_features=embed_dim)
		self.aggregate = Conv2d_BN(embed_dim * (num_path + 1),
								   out_embed_dim,
								   act_layer=nn.Hardswish)

	def forward(self, inputs):
		"""foward function"""
		att_outputs = [self.InvRes(inputs[0])]
		for x, encoder in zip(inputs, self.mhca_blks):
			# [B, C, H, W] -> [B, N, C]
			_, _, H, W = x.shape
			y = x.flatten(2).transpose(1, 2)
			y = encoder(y, size=(H, W))
			att_outputs.append(y)
			# print('MHCA:', x.shape, y.shape)
		out_concat = torch.cat(att_outputs, dim=1)
		out = self.aggregate(out_concat)
		return out

class Cls_head(nn.Module):
	"""a linear layer for classification."""
	def __init__(self, embed_dim, num_classes):
		"""initialization"""
		super().__init__()

		self.cls = nn.Linear(embed_dim, num_classes)

	def forward(self, x):
		"""foward function"""
		# (B, C, H, W) -> (B, C, 1)

		x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
		# Shape : [B, C]
		out = self.cls(x)
		return out

def dpr_generator(drop_path_rate, num_layers, num_stages):
	"""Generate drop path rate list following linear decay rule."""
	dpr_list = [
		x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
	]
	dpr = []
	cur = 0
	for i in range(num_stages):
		dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
		dpr.append(dpr_per_stage)
		cur += num_layers[i]

	return dpr

class MPViT(nn.Module):
	"""Multi-Path ViT class."""
	__name__='mpvit'
	def __init__(
		self,
		num_stages=4,
		num_path=[4, 4, 4, 4],
		num_layers=[1, 1, 1, 1],
		embed_dims=[64, 128, 256, 512],
		mlp_ratios=[8, 8, 4, 4],
		num_heads=[8, 8, 8, 8],
		drop_path_rate=0.1,
		in_chans=3,
		num_classes=1000,
		**kwargs,
	):
		super().__init__()

		self.num_classes = num_classes
		self.num_stages = num_stages
		self.embed_dims = embed_dims

		dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

		self.stem = nn.Sequential(
			Conv2d_BN(
				in_chans,
				embed_dims[0] // 2,
				kernel_size=3,
				stride=2,
				pad=1,
				act_layer=nn.Hardswish,
			),
			Conv2d_BN(
				embed_dims[0] // 2,
				embed_dims[0],
				kernel_size=3,
				stride=1,
				pad=1,
				act_layer=nn.Hardswish,
			),
		)

		# Patch embeddings.
		self.patch_embed_stages = nn.ModuleList([
			Patch_Embed_stage(
				embed_dims[idx],
				num_path=num_path[idx],
				isPool=False if idx == 0 else True,
			) for idx in range(self.num_stages)
		])

		# Multi-Head Convolutional Self-Attention (MHCA)
		self.mhca_stages = nn.ModuleList([
			MHCA_stage(
				embed_dims[idx],
				embed_dims[idx + 1]
				if not (idx + 1) == self.num_stages else embed_dims[idx],
				num_layers[idx],
				num_heads[idx],
				mlp_ratios[idx],
				num_path[idx],
				drop_path_list=dpr[idx],
			) for idx in range(self.num_stages)
		])

		# Classification head.
		self.cls_head = Cls_head(embed_dims[-1], num_classes)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		"""initialization"""
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=0.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def get_classifier(self):
		"""get classifier function"""
		return self.head

	def forward_features(self, x):
		"""forward feature function"""
		# print('input:', x.shape)
		x = self.stem(x)  # Shape : [B, C, H/4, W/4]
		# print('stem:', x.shape)
		xs = []
		for idx in range(self.num_stages):
			att_inputs = self.patch_embed_stages[idx](x)
			x = self.mhca_stages[idx](att_inputs)
			# print('ViT:', idx, x.shape)
			# print(idx, att_inputs[-1].shape, x.shape)
			xs.append(x)
		return xs

	def forward(self, x):
		"""foward function"""
		xs = self.forward_features(x)

		# cls head
		out = self.cls_head(xs[-1])
		return out

# def mpvit_tiny(**kwargs):
# 	model = MPViT(
# 		num_stages=4,
# 		num_path=[1, 2, 2, 2],
# 		num_layers=[1, 2, 2, 2],
# 		embed_dims=[64, 96, 176, 216],
# 		mlp_ratios=[1, 1, 1, 1],
# 		num_heads=[8, 8, 8, 8],
# 		**kwargs,
# 	)
# 	return model
def mpvit_tiny(**kwargs):
	model = MPViT(
		num_stages=4,
		num_path=[1, 1, 1, 1],
		num_layers=[1, 1, 1, 1],
		embed_dims=[64, 96, 128, 160],
		mlp_ratios=[1, 1, 1, 1],
		num_heads=[4, 4, 4, 4],
		**kwargs,
	)
	return model

def mpvit_small(**kwargs):
	model = MPViT(
		num_stages=4,
		num_path=[2, 3, 3, 3],
		num_layers=[1, 3, 6, 3],
		embed_dims=[64, 128, 216, 288],
		mlp_ratios=[4, 4, 4, 4],
		num_heads=[8, 8, 8, 8],
		**kwargs,
	)
	return model

def mpvit_base(**kwargs):
	model = MPViT(
		num_stages=4,
		num_path=[2, 3, 3, 3],
		num_layers=[1, 3, 8, 3],
		embed_dims=[128, 224, 368, 480],
		mlp_ratios=[4, 4, 4, 4],
		num_heads=[8, 8, 8, 8],
		**kwargs,
	)
	return model

class MSCA(nn.Module):#SegNeXt Attention
	def __init__(self, dim=1):
		super(MSCA, self).__init__()
		self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
		self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
		self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

		self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
		self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

		self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
		self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
		self.conv3 = nn.Conv2d(dim, dim, 1)

	def forward(self, x, size=None):
		# x = x.unsqueeze(1)
		u = x.clone()
		attn = self.conv0(x)

		attn_0 = self.conv0_1(attn)
		attn_0 = self.conv0_2(attn_0)

		attn_1 = self.conv1_1(attn)
		attn_1 = self.conv1_2(attn_1)

		attn_2 = self.conv2_1(attn)
		attn_2 = self.conv2_2(attn_2)
		attn = attn + attn_0 + attn_1 + attn_2

		attn = self.conv3(attn) * u

		return attn#.squeeze(1)

class CrossCNNBlock(torch.nn.Module):
	def __init__(self, in_c, out_c=1, ksize=9, shortcut=False):
		super(CrossCNNBlock, self).__init__()
		# self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))

		self.block12 = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
			nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
			nn.LeakyReLU(),nn.BatchNorm2d(out_c),
		)
		self.block34 = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size=(1,ksize), padding=(0,ksize//2)),
			nn.Conv2d(out_c, out_c, kernel_size=(ksize,1), padding=(ksize//2,0)),
			nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
			nn.LeakyReLU(),nn.BatchNorm2d(out_c),
		)
		self.block5 = nn.Sequential(
			# nn.Dropout2d(p=0.15),
			nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
			nn.LeakyReLU(),nn.BatchNorm2d(out_c),
		)
		# self.att = MSCA(out_c)
	def forward(self, x):
		out = F.gelu(self.block12(x) + self.block34(x))
		# out = self.att(out)
		return self.block5(out)#+x# + self.shortcut(x)

class CrossResNet(nn.Module):
	__name__='crnet'
	def __init__(self, in_ch=3, out_ch=6, flag_tiny=False):
		super(CrossResNet, self).__init__()
		if flag_tiny:
			layers = (32,32,32,32,32)
		else:
			layers = (32,64,96,128,256)
		self.layer_dims = layers
		ksizes = [13,11,9,7,5]
		self.pool = nn.MaxPool2d(kernel_size=2)
		self.path_estan = nn.ModuleList([CrossCNNBlock(in_c=layers[0], out_c=layers[0], ksize=ksizes[0]),])
		for i in range(len(layers)-1):
			self.path_estan.append(CrossCNNBlock(in_c=layers[i], out_c=layers[i + 1], ksize=ksizes[i+1]))

		self.cnn = nn.Sequential(
				nn.Conv2d(3, layers[0], kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(layers[0]),
				# nn.LeakyReLU()
			)
	def forward(self, x):
		xs = []
		x = self.cnn(x)
		for i, enc in enumerate(self.path_estan):
			x = enc(x)
			xs.append(x)
			x = self.pool(x)
			# print('CNN:', i, x.shape)
		return xs

class MPUpBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(MPUpBlock, self).__init__()
		self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
		self.prep = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.LeakyReLU(inplace=True)
			)
		self.post = nn.Sequential(
			nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
			# nn.BatchNorm2d(out_ch),
			# nn.LeakyReLU(inplace=True)
			)
		# self.ham = Hamburger(out_ch)

	def forward(self, x1, x2=None):
		"""
		Args:
			x1: [b,c, h, w]
			x2: [b,c, 2*h,2*w]
		Returns: 2x upsampled double conv reselt
		"""
		x1 = self.prep(x1)
		x1 = self.up(x1)
		if x2 is not None:
			x1 = x1 + x2
		x1 = self.post(x1)
		return x1#self.ham(x1)

class GateFusion(nn.Module):
	def __init__(self):
		super(GateFusion, self).__init__()
		self.relu = nn.LeakyReLU(inplace=True)

	def forward(self, x1, x2=None):
		if self.training:
			# alpha = random.random()
			B,C,H,W = x1.shape
			alpha = torch.rand(B,C,max(3,H//32),max(3,W//32))#*0.8+0.1
			alpha = F.interpolate(alpha, size=(H,W), mode='bicubic').to(x1.device)
			alpha = alpha.clamp(0,1)
		else:
			alpha = 0.5
		# print('alpha:', alpha.shape, x1.shape, x2.shape)
		# return self.relu(torch.sigmoid(x1)*x2*alpha + torch.sigmoid(x2)*x1*(1-alpha))
		return x1*alpha + x2*(1-alpha)

def SimpleFusion(x1, x2):
	return x1+x2

def norm_add(xs):
	# for x in xs:
	# 	print(x.shape)
	xs = [F.normalize(x, dim=1, p=2) for x in xs]
	xs = [F.interpolate(x, size=xs[0].shape[-2:], mode='bilinear', align_corners=False) for x in xs]
	return [sum(xs)/len(xs)]

class FTC(nn.Module):
	__name__ = 'gtc'
	def __init__(self, base_cnn, base_vit, out_channels=5, filters=32, flag_gate=True, flag_cnn=True, flag_vit=True, **args):
		super().__init__()
		self.flag_cnn = flag_cnn
		self.flag_vit = flag_vit

		self.base_vit = base_vit#mpvit_tiny()
		embed_dims = self.base_vit.embed_dims
		print('DIMS-VIT:', embed_dims)

		self.base_cnn = base_cnn#CrossResNet()
		layer_dims = self.base_cnn.layer_dims
		print('DIMS-CNN:', layer_dims)
		print('CHES-NET:', out_channels)

		self.tran_vit0 = nn.Sequential(nn.Conv2d(embed_dims[1], layer_dims[1], 1,1,0),nn.BatchNorm2d(layer_dims[1]))
		self.tran_vit1 = nn.Sequential(nn.Conv2d(embed_dims[2], layer_dims[2], 1,1,0),nn.BatchNorm2d(layer_dims[2]))
		self.tran_vit2 = nn.Sequential(nn.Conv2d(embed_dims[3], layer_dims[3], 1,1,0),nn.BatchNorm2d(layer_dims[3]))
		self.tran_vit3 = nn.Sequential(nn.Conv2d(embed_dims[3], layer_dims[4], 1,1,0),nn.BatchNorm2d(layer_dims[4]))

		self.tran_cnn0 = nn.Sequential(nn.Conv2d(layer_dims[1], layer_dims[1], 1,1,0),nn.BatchNorm2d(layer_dims[1]))
		self.tran_cnn1 = nn.Sequential(nn.Conv2d(layer_dims[2], layer_dims[2], 1,1,0),nn.BatchNorm2d(layer_dims[2]))
		self.tran_cnn2 = nn.Sequential(nn.Conv2d(layer_dims[3], layer_dims[3], 1,1,0),nn.BatchNorm2d(layer_dims[3]))
		self.tran_cnn3 = nn.Sequential(nn.Conv2d(layer_dims[4], layer_dims[4], 1,1,0),nn.BatchNorm2d(layer_dims[4]))
		self.gate = GateFusion() if flag_gate else SimpleFusion

		self.head = nn.Sequential(
				nn.Conv2d(layer_dims[-1], layer_dims[-1], kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(layer_dims[-1]),
				nn.LeakyReLU()
			)
		self.fuse = nn.Conv2d(layer_dims[4], 32, kernel_size=1)

		self.dec1 = MPUpBlock(layer_dims[-1], layer_dims[-2])
		self.dec2 = MPUpBlock(layer_dims[-2], layer_dims[-3])
		self.dec3 = MPUpBlock(layer_dims[-3], layer_dims[-4])
		self.dec4 = MPUpBlock(layer_dims[-4], filters)

		self.aux0 = nn.Conv2d(filters, out_channels, kernel_size=1)
		self.aux1 = nn.Conv2d(layer_dims[-4], out_channels, kernel_size=1)
		self.aux2 = nn.Conv2d(layer_dims[-3], out_channels, kernel_size=1)
		self.aux4 = nn.Conv2d(layer_dims[-2], out_channels, kernel_size=1)

	def forward(self, x):
		# Extracting features from CNN and ViT
		xs = self.base_cnn(x)
		c1,c2,c3,c4,c5 = xs
		xs = self.base_vit.forward_features(x)
		x2,x3,x4,x5 = xs
		# print('tran:', c2.shape, x2.shape)
		# print('ViT:', x2.shape, x3.shape,x4.shape,x5.shape)
		# print('CNN:', c2.shape, c3.shape,c4.shape,c5.shape)

		# Fusion features of CNN and ViT		
		if self.flag_vit and self.flag_cnn:
			x1 = c1
			x2 = self.gate(self.tran_vit0(x2), self.tran_cnn0(c2))
			x3 = self.gate(self.tran_vit1(x3), self.tran_cnn1(c3))
			x4 = self.gate(self.tran_vit2(x4), self.tran_cnn2(c4))
			x5 = self.gate(self.tran_vit3(x5), self.tran_cnn3(c5))
		elif self.flag_cnn:
			x1,x2,x3,x4,x5 = c1,c2,c3,c4,c5
		elif self.flag_vit:
			x1,x2,x3,x4,x5 = c1,self.tran_vit0(x2),self.tran_vit1(x3),self.tran_vit2(x4),self.tran_vit3(x5)

		# print(x.shape, x2.shape, x3.shape,x4.shape,x5.shape)
		# Decoder
		y8 = self.head(x5)
		y4 = self.dec1(y8, x4)  # 256,16,16
		# print(y.shape, x3.shape)
		y2 = self.dec2(y4, x3)
		y1 = self.dec3(y2, x2)
		y0 = self.dec4(y1, c1)
		
		self.feats = norm_add([x1,x2,x3,y0,y1,y2])
		# self.feat = y0.clone()
		# self.feats = self.feats + norm_add([y0,y1,y2,y4])

		# Output
		y0 = self.aux0(y0)
		y1 = F.interpolate(self.aux1(y1), size=x.shape[-2:], mode='bilinear', align_corners=False)
		y2 = F.interpolate(self.aux2(y2), size=x.shape[-2:], mode='bilinear', align_corners=False)
		y4 = F.interpolate(self.aux4(y4), size=x.shape[-2:], mode='bilinear', align_corners=False)
		# print("DS:", y0.shape, y1.shape, y2.shape, y4.shape)
		return [y0,y1,y2,y4]
		# return torch.softmax(self.final(y), dim=1)

######################################################Gated Fusion
def gtc_tt(n_class=8, **args):
	base_vit = mpvit_tiny()
	base_cnn = CrossResNet(flag_tiny=True)
	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=True)
	model.__name__ = 'gtctt'
	return model
def gtc_tb(n_class=8, **args):
	base_vit = mpvit_tiny()
	base_cnn = CrossResNet(flag_tiny=False)
	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=True)
	model.__name__ = 'gtctb'
	return model
	
# def gtc_st(n_class=8, **args):
# 	base_vit = mpvit_small()
# 	base_cnn = CrossResNet(flag_tiny=True)
# 	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=True)
# 	model.__name__ = 'gtcst'
# 	return model
# def gtc_sb(n_class=8, **args):
# 	base_vit = mpvit_small()
# 	base_cnn = CrossResNet(flag_tiny=False)
# 	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=True)
# 	model.__name__ = 'gtcsb'
# 	return model

# def gtc_bt(n_class=8, **args):
# 	base_vit = mpvit_base()
# 	base_cnn = CrossResNet(flag_tiny=True)
# 	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=True)
# 	model.__name__ = 'gtcbt'
# 	return model
# def gtc_bb(n_class=8, **args):
# 	base_vit = mpvit_base()
# 	base_cnn = CrossResNet(flag_tiny=False)
# 	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=True)
# 	model.__name__ = 'gtcbb'
# 	return model
	
######################################################Simple Fusion
def stc_tt(n_class=8, **args):
	base_vit = mpvit_tiny()
	base_cnn = CrossResNet(flag_tiny=True)
	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=False)
	model.__name__ = 'stctt'
	return model
tcct = stc_tt
def stc_tb(n_class=8, **args):
	base_vit = mpvit_tiny()
	base_cnn = CrossResNet(flag_tiny=False)
	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=False)
	model.__name__ = 'stctb'
	return model
	
# def stc_st(n_class=8, **args):
# 	base_vit = mpvit_small()
# 	base_cnn = CrossResNet(flag_tiny=True)
# 	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=False)
# 	model.__name__ = 'stcst'
# 	return model
# def stc_sb(n_class=8, **args):
# 	base_vit = mpvit_small()
# 	base_cnn = CrossResNet(flag_tiny=False)
# 	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=False)
# 	model.__name__ = 'stcsb'
# 	return model

def cnnu(n_class=8, **args):
	base_vit = mpvit_tiny()
	base_cnn = CrossResNet(flag_tiny=True)
	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=False, flag_vit=False, flag_cnn=True)
	model.__name__ = 'cnnu'
	return model

def vitu(n_class=8, **args):
	base_vit = mpvit_tiny()
	base_cnn = CrossResNet(flag_tiny=True)
	model = FTC(base_vit=base_vit, base_cnn=base_cnn, out_channels=n_class, flag_gate=False, flag_vit=True, flag_cnn=False)
	model.__name__ = 'vitu'
	return model

def points_selection_bins(feat, prob, true, card=512, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[1]
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	prob = prob[true>.5].view(-1)
	# print('ft:', feat.shape, feat.min().item(), feat.max().item())
	# with torch.no_grad():
	seq, idx = torch.sort(prob, dim=-1, descending=True)#降序排列
	# print('seq:', seq)	#seq: tensor([0.2426, 0.2354, 0.2335,  ..., 0.0633, 0.0612, 0.0608])
	# print('sort:', seq.shape, idx.shape, feat.shape, prob.shape)
	bins = 32
	N = feat.shape[0]//bins
	fs = []
	for i in range(bins):
		ni = N*i
		nj = ni+N
		f = torch.index_select(feat, dim=0, index=idx[ni:nj])
		# print('lh:', l.shape, h.shape, N, i)
		# print(h.shape, l.shape)
		f = f.mean(dim=0, keepdim=True)
		fs.append(f)
	f = torch.cat(fs, dim=0)
	# print('lh:', l.shape, h.shape, l.max().item(), h.max().item())
	# print(prob[idx[:card]].view(-1)[:9])
	# print(prob[idx[-card:]].view(-1)[:9])
	return f


class RegNet(nn.Module):
	__name__ = 'reg'
	tmp={}
	def __init__(self, base, out_channels=5, con='cor', num_emb=32):#(32,64,96,128,160)
		super(RegNet, self).__init__()
		self.base = base
		self.__name__ = base.__name__
		self.out_channels = out_channels

		self.lap_epl = nn.Sequential(
			nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1),
			# nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid()
		)
		dim_reg = out_channels-1
		self.lap_reg = nn.Sequential(
			nn.Conv2d(dim_reg, dim_reg, kernel_size=3, stride=1, padding=1, groups=dim_reg),
			nn.Conv2d(dim_reg, dim_reg, kernel_size=3, stride=1, padding=1, groups=dim_reg),
			# nn.Conv2d(dim_reg, dim_reg, kernel_size=3, stride=1, padding=1, groups=dim_reg),
			# nn.Conv2d(dim_reg, 1, kernel_size=3, stride=1, padding=1, groups=1),
		)
		self.lap_map = nn.Sequential(
			nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1),
			nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1),
			nn.Sigmoid()
		)
		self.tau = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32)*100)

	def forward(self, x):
		y = self.base(x)
		return y



if __name__ == '__main__':

	net = stc_tt(8)
	net = RegNet(net, out_channels=8)
	path = r'task1\onnx\tcct_heg.pt'
	pt = torch.load(path, map_location='cpu')
	print(net.load_state_dict(pt, strict=False))
	print('loaded model:', path)

	x = torch.rand(2,3,64,128)
	y = net(x)[0]
	print(y.shape)
