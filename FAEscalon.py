from mxnet import gluon, np, npx, nd
from mxnet.gluon import nn

npx.set_np()

class FAEscalon(nn.Block):
	def __init__(self, **kwargs):
		super(FAEscalon, self).__init__(**kwargs)

	def forward(self, data):
		aux = []
		i = 0
		for x in data:
			print(f'x{i+1}->',x)
			aux.append(0 if x < 1 else 1)
			i += 1
		print(aux)
		return nd.array(aux)