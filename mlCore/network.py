
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from smodels.tools.smodelsLogging import logger

from scipy.special import inv_boxcox
#from sklearn.preprocessing import MinMaxScaler

def getNodesPerLayer(shape, nodes, layer, fullDim):

	"""
	Translates shape parameter into nodes per layer

	:param shape: (str) shape of the model: trap, lin, ramp
	:param nodes: (int) number of nodes in largest layer
	:param layer: (int) number of hidden layer
	:param fullDim: (int) number of mass inputs

	"""

	net = []
	nodes_total = 0
	
	for lay in range(layer):

		n = [0, 0]
		n_count = 0

		if shape == "lin":

			n[0] = nodes
			n[1] = nodes
			n_count += nodes

		elif shape == "trap":

			k = 2 * nodes / layer
			m = layer*0.5
			
			for i in range(2):

				cl = float(lay + i)
			
				if cl > m:
					cl = m - (cl%m)
				
				n[i] = round(cl*k)

			n_count += n[i]

		elif shape == "ramp":
			
			k = nodes / layer
	
			for i in range(2):
	
				cl = float(lay + i - 1)
				n[i] = round(nodes - k * cl)
	
			if lay == 0:
				n[1] = nodes
			elif lay == 1:
				n[0] = nodes

			n_count += n[i]				

		if lay == 0:
			n[0] = fullDim
		if lay == layer - 1:
			n[1] = 1
			n_count = 0

		nodes_total += n_count
		net.append(n)

	return [net, nodes_total]


class DatabaseNetwork(nn.Module):

	"""
	Ensemble module that contains both regression and classification network. This is the final product of the machine learning system and will be stored in the database.

	"""

	def __init__(self, winner):

		"""
		Takes both best performing models of one analysis and stores them.

		:param winner: dict or both 'regression' and 'classification' models (dict:torch.nn.Module)

		"""
    	
		super(DatabaseNetwork, self).__init__()
		self["regression"] = winner["regression"]["model"]
		self["classification"] = winner["classification"]["model"]

		self["regression"].trainingMode = False
		self["classification"].trainingMode = False


	def __setitem__(self, netType, model):
		self.__dict__[netType] = model

	def __getitem__(self, netType):
		return self.__dict__[netType]

	def __repr__(self):
		return repr(self.__dict__)

	def __len__(self):
		return len(self.__dict__)

	def getValidationLoss(self, netType):
		return self[netType].getValidationLoss()

	def setSpeedFactor(self, factor):
		self._speedFactor = factor

	def getSpeedFactor(self):
		return self._speedFactor

	def forward(self, x):

		"""
		Main method that is called with model(input)
		Sends the input masses to the classification network. If the classifier outputs 1 (on hull) 
		the masses get sent to the regression model for a target prediction

		:param x: (tensor) input masses

		"""
		
		if self["classification"] == None:
			onHull = True
		else:
			onHull = self["classification"](x) == 1.

		if onHull:
			target = self["regression"](x)
			return target

		return None


	def save(self, expres, txNameData):

		dbPath = expres.path
		for i in range(len(dbPath)):
			if dbPath[i:i+8] == 'database':
				dbPath = dbPath[i:]
				break
		path = os.getcwd() + "/" + dbPath + "/models"
		Path(path).mkdir(parents=True, exist_ok=True)
		path += "/" + str(txNameData) + ".pth"

		torch.save(self, path)
		logger.info("model saved at '%s'" % path)


	def load(expres, txNameData):
	
		dbPath = expres.path
		for i in range(len(dbPath)):
			if dbPath[i:i+8] == 'database':
				dbPath = dbPath[i:]
				break
		path = os.getcwd() + "/" + dbPath + "/models/" + str(txNameData) + ".pth"

		try:
			model = torch.load(path)
			model.eval()
		except: model = None

		return model


class Net_cla(nn.Module):

	"""
	Classification network

	"""
 
	def __init__(self, netShape, activFunc):

		super(Net_cla, self).__init__()
		self.seq = nn.Sequential()
		self.trainingMode = True
		self._delimiter = 0.
		lastLayer = len(netShape) - 1

		for i in range(len(netShape)):

			nin, nout = netShape[i][0], netShape[i][1]

			self.seq.add_module('lin{}'.format(i), nn.Linear(nin,nout))

			if activFunc == "rel" and i != lastLayer:
				self.seq.add_module('rel{}'.format(i), nn.ReLU()) #nn.BatchNorm1d(nout))

			if activFunc == "prel" and i != lastLayer:
				self.seq.add_module('prel{}'.format(i), nn.PReLU())

			if activFunc == "sel" and i != lastLayer:
				self.seq.add_module('sel{}'.format(i), nn.SELU())

			if activFunc == "lrel" and i != lastLayer:
				self.seq.add_module('lrel{}'.format(i), nn.LeakyReLU())

			if i == lastLayer:
				self.seq.add_module('sgm{}'.format(i), nn.Sigmoid())
			#elif i == 0:
				#self.seq.add_module('drp{}'.format(i), nn.Dropout(0.2))			


	def setValidationLoss(self, meanError):
		self._validationLoss = meanError

	def getValidationLoss(self):
		return self._validationLoss

	def setRescaleParameter(self, parameter):
		self._rescaleParameter = parameter


	def forward(self, x):#input_):

		"""
		Main method that is called with model(input)
		Rescaling parameters are saved during training and should never be changed afterwards
		Output is either (scaled) torch.tensor or unscaled np.array depending on whether model is in self.trainingMode

		:param x: (tensor) input masses

		"""

		if "_rescaleParameter" in self.__dict__:
			method = self._rescaleParameter["masses"]["method"]
		else:
			method = None

		if not self.trainingMode and method != None:

			x = x.detach().numpy()

			if method == "minmaxScaler":

				x = [x] # <--- SKETCHY
				scaler = self._rescaleParameter["masses"]["scaler"]
				x = scaler.transform(x)

			elif method == "standardScore":

				mean = self._rescaleParameter["masses"]["mean"]
				std = self._rescaleParameter["masses"]["std"]
				x = (x - mean) / std

			x = torch.tensor(x, dtype=torch.float64)

		x = self.seq(x)
		
		if not self.trainingMode and self._delimiter != 0.:
			for n in range(len(x)):
				if self._delimiter < x[n]: x[n] = 1.
				else: x[n] = 0.
		
		return x


class Net_reg(nn.Module):

	"""
	Regression network

	"""

	def __init__(self, netShape, activFunc):
    	
		super(Net_reg, self).__init__()
		self.seq = nn.Sequential()
		self.trainingMode = True

		lastLayer = len(netShape) - 1

		for i in range(len(netShape)):

			nin, nout = netShape[i][0], netShape[i][1]

			self.seq.add_module('lin{}'.format(i), nn.Linear(nin,nout))

			if activFunc == "rel" and i != lastLayer:
				self.seq.add_module('rel{}'.format(i), nn.ReLU())
                    
			if activFunc == "prel" and i != lastLayer:
				self.seq.add_module('prel{}'.format(i), nn.PReLU())

			if activFunc == "sel" and i != lastLayer:
				self.seq.add_module('sel{}'.format(i), nn.SELU())

			if activFunc == "lrel" and i != lastLayer:
				self.seq.add_module('lrel{}'.format(i), nn.LeakyReLU()) 

    	

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)


	def setValidationLoss(self, meanError):
		self._validationLoss = meanError

	def getValidationLoss(self):
		return self._validationLoss

	def setRescaleParameter(self, parameter):
		self._rescaleParameter = parameter



	def forward(self, x):

		"""
		Main method that is called with model(input)
		Rescaling parameters are saved during training and should never be changed afterwards
		Output is either (scaled) torch.tensor or unscaled np.array depending on whether model is in self.trainingMode

		:param x: (tensor) input masses

		"""

		if "_rescaleParameter" in self.__dict__:
			method = self._rescaleParameter["masses"]["method"]
		else:
			method = None

		if not self.training and method != None:

			x = x.detach().numpy()

			if method == "minmaxScaler":

				scaler = self._rescaleParameter["masses"]["scaler"]
				x = scaler.transform(x)

			elif method == "standardScore":

				mean = self._rescaleParameter["masses"]["mean"]
				std = self._rescaleParameter["masses"]["std"]
				x = (x - mean) / std

			x = torch.tensor(x, dtype=torch.float64)

		x = self.seq(x)

		if "_rescaleParameter" in self.__dict__:
			method = self._rescaleParameter["targets"]["method"]
		else:
			method = None

		if not self.trainingMode: # and method != None:

			x = x.detach().numpy()

			if method == "boxcox":

				lmbda = self._rescaleParameter["targets"]["lambda"]
				x = [inv_boxcox(t, lmbda)[0] for t in x]

			elif method == "log":

				x = [(10**t)[0] for t in x]

			elif method == "standardScore":

				mean = self._rescaleParameter["targets"]["mean"]
				std = self._rescaleParameter["targets"]["std"]
				x = x * std + mean

			else:

				x = [t[0] for t in x]

		return x
	


def createNet(hyper, rescaleParameter, full_dim, nettype):

	"""
	Translates parameter instruction strings into actual models for training.
	This method is partially outdated and will be reworked for the final push

	"""

	shape = hyper["shape"]
	nodes = hyper["nodes"]
	layer = hyper["layer"]
	activ = hyper["activationFunction"]

	netshape, nodesTotal = getNodesPerLayer(shape, nodes, layer, full_dim)

	if nettype == 'regression':
		model = Net_reg(netshape, activ)
	elif nettype == 'classification':
		model = Net_cla(netshape, activ)
	
	return model

