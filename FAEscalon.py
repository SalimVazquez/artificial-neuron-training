from mxnet import gluon, nd
from mxnet.gluon import nn

class FAEscalon(nn.Block):
	def __init__(self, **kwargs):
		super(FAEscalon, self).__init__(**kwargs)

	def forward(self, x):
		return 0 if x <= 0 else 1