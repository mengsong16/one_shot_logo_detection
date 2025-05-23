
import sys
sys.path.append("./stn_pytorch")

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.optim
import math
import os
#from stn_pytorch.modules.stn import STN
#from stn_pytorch.modules.gridgen import CylinderGridGen, AffineGridGen, AffineGridGenV2

#If a module is run as a script, its __name__ is set to __main__ from the beginning


cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# make the feature extractor you would like to use
def make_layers(cfg, in_channels, batch_norm=False):
	layers = []
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)	


# a network to extract features
class Feature_Extractor(nn.Module):
	def __init__(self):
		super(Feature_Extractor, self).__init__()
		
		self.features = make_layers(in_channels=3, cfg=cfg['VGG16'])

		# classifier
		self.classifier = nn.Sequential(
			# fc6
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			# fc7
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout()
		)
		
		self._initialize_weights()

	def forward(self, input_image):
		######## will stuck here if gpu parallization fails !!!
		feas = self.features(input_image)
		# resize to [batch_size, n]
		feas = feas.view(feas.size(0), -1)
		# out is 4096 fc7 vector
		out = self.classifier(feas)
		
		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


# create model			
def create_model(pretrain_path=""):
	print("Model name: VGG16")
	# create model
	model = Feature_Extractor()
	print('==> Creating model from pretrained model: '+pretrain_path)
	print('==> Loading pretrained model...')
	model = load_matched_model(model, pretrain_path)
		
	return model

# load matched model
def load_matched_model(model, pretrain_path):
	pretrained_dict = torch.load(pretrain_path)
	model_dict = model.state_dict()
	model_size = len(model_dict)
	print('model size: '+ str(model_size))
	print('pretrained model size: '+ str(len(pretrained_dict)))

	# 1. filter out unnecessary keys
	# model has _parameters which inclues items [name, param]
	# state_dict() is a dictionary which includes items [prefix + name, param.data]
	# k: name, v: parameters		
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size()==v.size()}
	cur_len = len(pretrained_dict)	
		
	# 2. overwrite entries in the existing state dict
	# if some key in pretrained_dict do no exist in model_dict, error will be raised
	# matching is done by name	
	model_dict.update(pretrained_dict)
	
	# 3. load the new state dict
	# update all parameters and buffers in the state_dict of model
	model.load_state_dict(model_dict)

	
	# print loaded and not loaded layers
	print('[model_dict] Matched layers: '+ str(cur_len)+' / '+str(model_size))
	for k, v in pretrained_dict.items():
		print("	"+str(k)+": "+str(v.size()))
	
	return model		

# main
if __name__ == '__main__':
	root_dir = "/work/meng"
	#net = vgg16_stack(num_classes=2, pretrained=True, pretrain_path="/home/meng/Documents/pytorch/uvn/models/vgg16.pth")
	#net, optimizer = create_model_optimizer(model_name="vgg16_stack", pretrained=False, base_lr=0.1, momentum=0.9, weight_decay=5e-4)
	net = create_model(pretrain_path=os.path.join(root_dir, "uvn/pretrained/vgg16.pth"))
	x = torch.randn(8,3,224,224)
	print(net(Variable(x)).size())
	'''
	pl = list(map(id, net.parameters()))
	pm = net.state_dict()
	pn = [name for name, param in net.state_dict().items()]
	pn_c = [name for name, param in net.classifier.state_dict().items()]
	print(pn_c)
	'''
