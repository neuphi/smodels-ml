#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
from math import ceil, inf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from parameterParser import Parameter
from sklearn.preprocessing import MinMaxScaler
from mlCore.network import DatabaseNetwork
from mlCore.dataset import DatasetBuilder
from mlCore.lossFunctionsAndOptimizers import getModelError
from smodels.tools.physicsUnits import GeV, fb
from smodels.theory.auxiliaryFunctions import unscaleWidth
from smodels.tools.smodelsLogging import logger

class NetworkEvaluater():

	def __init__(self, parameter, dataset, builder, model = None):

		"""


		"""

		self.expres = parameter["expres"]
		self.txnameData = parameter["txName"].txnameData
		self.dataselector = parameter["dataselector"]
		self.nettype = parameter["nettype"]
		self.massColumns = parameter["massColumns"]

		if model != None:
			self.model = model
		else:
			self.model = DatabaseNetwork.load(self.expres, self.txnameData) #loadModel(expres, txName)[netType]#.double()

		dbPath = self.expres.path
		for i in range(len(dbPath)):
			if dbPath[i:i+8] == 'database':
				dbPath = dbPath[i:]
				break
		self.savePath = os.getcwd() + "/" + dbPath + "/performance/"
		Path(self.savePath).mkdir(parents=True, exist_ok=True)

		self.builder = builder
		self.dataset = dataset
		#self.unscaleData()


	"""
	def unscaleData(self, showPlots = False):

		model = self.model["regression"]
		predictions = model(self.dataset.inputs).detach().numpy()
		scaler = model.scaler
		#s1, s2 = [], []

		self.inputsRaw = scaler["masses"].inverse_transform(self.dataset.inputs)
		#self.labelsRaw = scaler["targets"].inverse_transform(self.dataset.labels)
		#self.predicRaw = scaler["targets"].inverse_transform(predictions)#np.reshape(predictions, (-1, 1)))
		self.labelsRaw = self.dataset.labels.detach().numpy()
		self.predicRaw = predictions

		from scipy.special import inv_boxcox
		#lmbda = -0.02888445 #no 0
		#lmbda = 0.12332622

		self.labelsRaw = [inv_boxcox(-label[0], self.lmbda) for label in self.labelsRaw]
		self.predicRaw = [inv_boxcox(-predic[0], self.lmbda) for predic in self.predicRaw]

		'''
		for n in range(len(predictions)):
			if self.dataset.inputs[n][0] == 1 and self.dataset.inputs[n][2] < 73.841:
				print(self.dataset.labels[n], predictions[n])
				print(self.labelsRaw[n], self.predicRaw[n])
				print("---")
		'''
		
		
		#self.labelsRaw = inv_boxcox(self.labelsRaw, lmbda)
		#self.predicRaw = inv_boxcox(self.predicRaw, lmbda)


		#self.labelsRaw = [0. if label > 70. else unscaleWidth(label[0]).asNumber(GeV) for label in self.labelsRaw]
		#self.predicRaw = [0. if predic > 70. else unscaleWidth(predic[0]).asNumber(GeV) for predic in self.predicRaw]

		## old
		##self.labelsRaw = [10**-label[0] for label in self.labelsRaw]
		##self.predicRaw = [10**-predic[0] for predic in self.predicRaw]
	"""

	def binError(self, whichData, showPlots = False):

		nettype = "regression"
		error = getModelError(self.model[nettype], self.dataset, nettype, returnMean = False)

		if whichData == "labels":
			data = self.labelsRaw
			units = ""
		elif whichData == "widths":
			data = []
			units = "GeV"
			for iR in self.inputsRaw:
				data.append(unscaleWidth(iR[2]).asNumber(GeV)) #widthPos]

		log = []
		for d in data:
			try: d = d.item()
			except: d = d
			if d != 0. and np.log10(d) != inf:
			#if d != 0.:
				log.append(np.log10(d))


		binMin = min(log)
		binMax = max(log)

		binNum = ceil(max(abs(binMin),abs(binMax))) + 1

		bins = [[] for _ in range(binNum)]
		mean = [_ for _ in range(binNum)]
		std  = [_ for _ in range(binNum)]

		for n,e in enumerate(error):
			try: d = data[n].item()
			except: d = data[n]
			if d == 0.: index = 0
			elif d == inf: index = 0 # WHAT IS GOING ON
			else: index = ceil(abs(np.log10(d)))
			
			bins[index].append(e)

		bins = np.array(bins)

		for n,b in enumerate(bins):

			if len(b) > 0:
				mean[n] = np.mean(b)
				std[n] = np.std(b)
			else:
				mean[n] = 0
				std[n] = 0

		labels = [_ for _ in range(binNum)]

		for n in range(binNum):
			
			if n == 0.:
				labels[n] = "0 " + units

			elif n == 1:
				labels[n] = ">1e-1 " + units

			else:
				labels[n] = "1e-" + str(n-1) + " - 1e-" + str(n) + " " + units

			labels[n] += " (n = {})".format(str(len(bins[n])))


		x = np.arange(len(bins))  # the label locations
		width = 0.5

		fig, ax = plt.subplots()
		rects = ax.bar(x, mean, width, yerr=std)


		ax.set_ylabel('mean relative error')
		ax.set_title('mean error binned by %s (n = %s)' % (whichData, len(self.dataset)))
		ax.set_xticks(x)
		ax.set_xticklabels(labels, rotation=45, rotation_mode="anchor", ha="right")
		#ax.legend()

		for rect in rects:
			height = round(rect.get_height(), 3)
			ax.annotate('{}'.format(height),
				xy=(rect.get_x() + rect.get_width() / 2, height), 
				xytext=(0, 3),  # 3 points vertical offset
				textcoords="offset points",
				ha='center', va='bottom')

		fig.tight_layout()
		if showPlots: plt.show()
		


	def regression(self, showPlots = False):

		"""


		"""

		model = self.model["regression"]
		predictions = model(self.dataset.inputs)#.detach().numpy()

		error = []

		if "lambda" in model._rescaleParameter["targets"]:
			from scipy.stats import boxcox
			L = model._rescaleParameter["targets"]["lambda"]
		else:
			L = None

		for n in range(len(self.dataset.labels)):


			l = self.dataset.labels[n].detach().item()
			p = predictions[n] #[0]

			# optional if you want to plot the errors of rescaled targets
			#if L != None:
			#	l = boxcox(l, L)
			#	p = boxcox(p, L)
			#	or 
			#	l = 10**l
			#	p = 10**p

			if False:#self.builder.refXsecs != None:
				#m0 = self.inputsRaw[n][0]

				m0 = self.dataset.inputs[n][0]
				xsec = self.builder._getRefXsec(m0)
				
				if self.builder.luminosity * xsec * l < 1e-2: l = 0
				if self.builder.luminosity * xsec * p < 1e-2: p = 0
					


			if l != 0:
				e = np.sqrt((( p - l ) / l)**2)

				error.append(e)
			else:

				error.append(p)

	
		
		#for n, e in enumerate(E):
		#	if not e < 100: 
		#		error[n] = 0

		#self.inputsRaw = self.dataset.inputs
		#self.labelsRaw = self.dataset.labels

		# sorting data so largest errors are added last to scatter plots
		argsort = np.argsort(error)
		error = np.array(error)[argsort]


		#yaxis, waxis = 1, 2 # M1b
		#yaxis, waxis = 2, 6 # M5
		#yaxis, waxis = 1, 4 # M8

		xaxis = 0
		yaxis = len(self.dataset.inputs[0]) - 1
		widthPlot = False

		if self.massColumns != None: # gridpoints read from external file

			yaxis = self.massColumns.index(max(self.massColumns[:-1]))
			lspPlot = yaxis > xaxis

			w = min(self.massColumns)
			widthPlot = w < 0
			waxis = self.massColumns.index(w)

		mother = np.array([inputs[0] for inputs in self.dataset.inputs])[argsort]
		logTargets = ( np.log10(max(self.dataset.labels)) - np.log10(min(self.dataset.labels)) ) > 4


		if logTargets:
			#EFF = np.array([target for target in self.labelsRaw])[argsort]
			#EFF = np.array([np.log10(labels) for labels in self.labelsRaw])[argsort]
			targets = np.array([0. if target == 0. else np.log10(target) for target in self.dataset.labels])[argsort]
		else:
			targets = self.dataset.labels.detach().numpy()[argsort]

		thingsToPlot = {}


		if yaxis != xaxis:
			lsp = np.array([inputs[yaxis] for inputs in self.dataset.inputs])[argsort]
			thingsToPlot["mother_lsp"] = {"xaxis": mother, 
										"xlabel": r"$m_{mother}$ (GeV)", 
										"yaxis": lsp, 
										"ylabel":  r"$m_{LSP}$ (GeV)", 
										"error": error}


		if widthPlot:
			widths = [unscaleWidth(inputs[waxis].item()).asNumber(GeV) for inputs in self.dataset.inputs]
			widths_log = np.array([-40. if width == 0. else np.log10(width) for width in widths])[argsort] #0.
			thingsToPlot["mother_width"] = {"xaxis": mother, 
										"xlabel": r"$m_{mother}$ (GeV)", 
										"yaxis": widths_log, 
										"ylabel": "width (log10)", 
										"error": error}

		
		
		if self.dataselector == "efficiencyMap":
			ylabel = "efficiencies"
		else:
			ylabel = "upper limits"

		if logTargets:
			ylabel += r" (log$_{10}$)"
		
		thingsToPlot["mother_target"] = {"xaxis": mother, 
									"xlabel": r"$m_{mother}$ (GeV)", 
									"yaxis": targets, 
									"ylabel":  ylabel, 
									"error": error}

		
		if False:
			# bundle all values of constant m0 mass #
			targetMass = [140., 160., 240., 1000., 1200., 1400., 1600., 2000.]
			for tmass in targetMass:

				spliceWID = []
				spliceEFF = []
				spliceERR = []

				for n,mass in enumerate(mother):
					if mass == tmass:
						spliceWID.append(WIDTH_LOG[n])
						spliceEFF.append(EFF[n])
						spliceERR.append(E[n])

				if len(spliceERR) > 0:
					key = "m0=" + str(int(tmass))
					thingsToPlot[key] = {"yaxis": spliceEFF, 
										"ylabel": "efficiencies (log)", 
										"xaxis": spliceWID, "xlabel":  
										"widths (log)", 
										"error": spliceERR, 
										"affix": r"$m_{HSCP}$ = " + str(int(tmass)) + " GeV "}
		
		
		index = 5
		for key, value in thingsToPlot.items():

			maxError = np.max(value["error"]) * 100.
			meanError = np.mean(value["error"]) * 100.

			vMax = min(1., 0.01*maxError)

			if not "affix" in value:
				affix = " "
			else: affix = value["affix"]

			plt.figure(index)
			plt.title("{} (regression)\n{}mean error: {:4.2f}% max error: {:4.2f}%".format(str(self.txnameData), affix, meanError, maxError), fontsize=14)
			plt.xlabel(value["xlabel"])
			plt.ylabel(value["ylabel"])
			plt.scatter(value["xaxis"], value["yaxis"], c=value["error"], cmap='rainbow', vmin=0, vmax=vMax)
			cbar = plt.colorbar()
			cbar.set_label('relative error', rotation=90)
			plt.tight_layout()
			fileName = str(self.txnameData) + "_regression_scatterPlot_" + key + ".png" #eps
			plt.savefig(self.savePath + fileName)
			index += 1

		if showPlots: plt.show()

		

		



	def classification(self, showPlots = False):

		"""


		"""

		model = self.model["classification"]

		delimiter = 0.5
		onHull_correct = []
		onHull_wrong = []
		offHull_correct = []
		offHull_wrong = []

		for i in range(len(self.dataset)):

			inputs = self.dataset.inputs[i]
			predi = model(inputs)
			label = self.dataset.labels[i]

			#print(predi, label)

			if label == 0.:
				if predi == 0.:
					offHull_correct.append(inputs)
				else:
					offHull_wrong.append(inputs)
			elif label == 1.:
				if predi == 1.:
					onHull_correct.append(inputs)
				else:
					onHull_wrong.append(inputs)

		onHull_correct_total = len(onHull_correct)
		onHull_wrong_total = len(onHull_wrong)
		offHull_correct_total = len(offHull_correct)
		offHull_wrong_total = len(offHull_wrong)
		onHull_total = onHull_correct_total + onHull_wrong_total
		offHull_total = offHull_correct_total + offHull_wrong_total
		samples_total = onHull_total + offHull_total

		error = round(100.*(1. - (onHull_correct_total + offHull_correct_total)/samples_total), 3)
		delim = round(model._delimiter, 3)

		onShell  = "%s / %s (%s%%)" % (onHull_correct_total, onHull_total, round(100.*onHull_correct_total/onHull_total, 3))
		offShell = "%s / %s (%s%%)" % (offHull_correct_total, offHull_total, round(100.*offHull_correct_total/offHull_total, 3))
		total    = "%s / %s (%s%%)" % (onHull_correct_total + offHull_correct_total, samples_total, error)

		print("onShell:   %s" %onShell)
		print("offShell:  %s" %offShell)
		print("total:     %s" %total)
		print("delimiter: %s" %delim)


		plt.figure(0)
		plt.title('{} (regression)\nerror: {}% (delimiter: {})'.format(str(self.txnameData), error, delim), fontsize=14)
		plt.xlabel('mass mother [GeV]')
		plt.ylabel('mass daughter [GeV]')

		x = 0
		y = 1


		#plt_cor_on = plt.scatter([np.exp(oH[x].item()) for oH in onHull_correct], [np.exp(oH[y].item()) for oH in onHull_correct], color = 'green')
		#plt_cor_off = plt.scatter([np.exp(oH[x].item()) for oH in offHull_correct], [np.exp(oH[y].item()) for oH in offHull_correct], color = 'blue')
		#plt_wrg_on = plt.scatter([np.exp(oH[x].item()) for oH in onHull_wrong], [np.exp(oH[y].item()) for oH in onHull_wrong], color = 'red')
		#plt_wrg_off = plt.scatter([np.exp(oH[x].item()) for oH in offHull_wrong], [np.exp(oH[y].item()) for oH in offHull_wrong], color = 'orange')

		plt_cor_on = plt.scatter([oH[x].item() for oH in onHull_correct], [oH[y].item() for oH in onHull_correct], color = 'green')
		plt_cor_off = plt.scatter([oH[x].item() for oH in offHull_correct], [oH[y].item() for oH in offHull_correct], color = 'blue')
		plt_wrg_on = plt.scatter([oH[x].item() for oH in onHull_wrong], [oH[y].item() for oH in onHull_wrong], color = 'red')
		plt_wrg_off = plt.scatter([oH[x].item() for oH in offHull_wrong], [oH[y].item() for oH in offHull_wrong], color = 'orange')

		plt.legend((plt_cor_on, plt_cor_off, plt_wrg_on, plt_wrg_off), ('on hull correct', 'off hull correct', 'should be on hull', 'should be off hull'), scatterpoints=1, loc='upper right', ncol=1, fontsize=8)

		fileName = str(self.txnameData) + "_classification_scatterPlot.eps"
		plt.savefig(self.savePath + fileName)

		if showPlots: plt.show()


def main(parameter, nettypes):

	# ----------------------------------------------------------------------------------- #
	# custom dictionary class that automatically permutates all possible map combinations #
	# ----------------------------------------------------------------------------------- #

	thingsToEvaluate = parameter["database"]

	# -------------------------------------------------------------------------------- #
	# loop over all analysis map combinations of the parameter file [database] section #
	# -------------------------------------------------------------------------------- #

	while(thingsToEvaluate.incrIndex):

		if not parameter.loadExpres(): continue

		builder = DatasetBuilder(parameter)

		for nettype in nettypes: #,"classification"]:
			parameter.set("nettype", nettype)

			#builder.sampleSize[nettype] = 100
			builder.createDataset(nettype)
			builder.shuffle()
			#builder.rescaleMasses()
			#builder.rescaleTargets()

			dataset = builder.getDataset(fullSet = True, splitSet = False, rescaleParams = False)


			validater = NetworkEvaluater(parameter, dataset["full"], builder)

			#validater.binError("labels", showPlots = True)
			#validater.binError("widths", showPlots = True)

			if nettype == "regression":
				validater.regression(showPlots = True)
			else:
				validater.classification(showPlots = True)

if __name__=='__main__':

	ap = argparse.ArgumentParser(description="Evaluates performance of generated neural networks")
	ap.add_argument('-p', '--parfile', 
			help='parameter file', default='nn_parameters.ini')
	ap.add_argument('-l', '--log', 
			help='specifying the level of verbosity (error, warning, info, debug)',
			default = 'info', type = str)
	ap.add_argument('-n', '--nettype', 
			help="which neural network to test ('regression', 'classification' or 'all')",
			default = 'regression', type = str)     
	args = ap.parse_args()

	if args.nettype == "all": nettypes = ["regression, classification"]
	else: nettypes = [args.nettype]

	parameter = Parameter(args.parfile, args.log)

	main(parameter, nettypes)

