#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: trainModel.py
   :synopsis: run gridsearch to train optimal neural networks for smodels-database maps
.. moduleauthor:: Philipp Neuhuber <ph.neuhuber@gmail.com>

"""

import os
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.optimize import minimize
from mlCore.dataset import Data
from mlCore.network import createNet
from mlCore.lossFunctionsAndOptimizers import loadLossFunction, loadOptimizer, getModelError
from smodels.tools.smodelsLogging import logger
from smodels.tools.smodelsLogging import getLogLevel


class ModelTrainer():


	def __init__(self, parameter, dataset):

		"""
		Example usage of neural network training methods.
		Reads the parameter file and trains networks for all
		maps it can find.

		:param parameter: Custom parameter dictionary holding all meta and hyper param informations
		:param dataset: dict of customized torch Dataset classes for training, testing and validation

		"""

		self.dataset = dataset
		self.expres = parameter["expres"]
		self.txnameData = parameter["txName"].txnameData
		self.type = parameter["nettype"]
		self.hyper = parameter[self.type]

		self.device = parameter["device"]
		self.cores = parameter["cores"]

		self.full_dim = self.txnameData.full_dimensionality
		self.rescaleParameter = dataset["rescaleParams"]

		self.outputPath = parameter["outputPath"]

		self.dataset 	= dataset["full"]
		self.training 	= dataset["training"]
		self.testing	= dataset["testing"]
		self.validation	= dataset["validation"]
		
		self.logData = []
		self.winner = {"error": 1e10}
		

	def run(self):

		"""
		Loops over all hyperparameter configurations 
		and keeps track of the best current model.

		:return winner: result dict for best performing model, containing torch model, error and hyperparam config

		"""

		self.hyper.resetIndex
		while(self.hyper.incrIndex):

			self.epochLoss = {"training":[], "testing":[]}

			logger.info("training with hyperparam config %d/%d .." %(self.hyper.index + 1, len(self.hyper)))

			self.runCurrentConfiguration()

			if self.winner["error"] > self.meanError:

				self.winner["model"]	 = deepcopy(self.model)
				self.winner["error"]   	 = self.meanError
				self.winner["config"]  	 = self.hyper[-1]
				self.winner["epochLoss"] = self.epochLoss
				#self.winner["logData"]  = self.logData

				self.winner["model"].setValidationLoss(self.winner["error"])

		self.winner["model"].setRescaleParameter(self.rescaleParameter)

		return self.winner



	def runCurrentConfiguration(self, secondRun = False):

		"""
		Parent method of actual training. Handles training differencies between
		regression and classification and keeps track of current model's error on
		the validation set.

		:param secondRun: rerun model training with subset of inaccurate dataset predictions

		"""

		self.model = createNet(self.hyper[-1], self.rescaleParameter, self.full_dim, self.type).double().to(self.device)
		self.trainModel()

		if self.type == "regression" and secondRun:
			subset = self._getWrongPredictions()
			logger.debug("subset length: %s" %len(subset))
			self.trainModel(training = subset)
		
		self.meanError = getModelError(self.model, self.validation, self.type)[0]

		if self.type == "classification":
			predictions, labels = self.model(self.validation.inputs).detach().numpy(), self.validation.labels.detach().numpy()
			self.model._delimiter = minimize(self._findDelimiter, 0.5, args=(predictions, labels), method="Powell").x.tolist()



	def trainModel(self, optimizer = None, lossFunction = None, batchSize = 0, epochNum = 0, training = None, testing = None):

		"""
		Core training method. Loads necessary torch classes and training parameters and
		updates the models' weights and biases via back propagation.

		"""

		if training == None:  	training	  = self.training
		if testing == None: 	 testing 	  = self.testing
		if batchSize == 0: 		 batchSize 	  = self.hyper["batchSize"]
		if epochNum == 0: 		 epochNum 	  = self.hyper["epochNum"]
		if optimizer == None: 	 optimizer 	  = loadOptimizer(self.hyper["optimizer"], self.model, self.hyper["learnRate"])
		if lossFunction == None: lossFunction = loadLossFunction(self.hyper["lossFunction"], self.device)

		lossFunc2 = loadLossFunction("MSE", self.device)
		alpha, beta = 1., 50.

		trainloader = DataLoader(training, batch_size = batchSize, shuffle = True, num_workers = self.cores)

		bestLossLocal, bestEpochLocal, bestModelLocal  = 1e6, 0, deepcopy(self.model)

		for epoch in range(epochNum):

			self.model.train()

			for i, data in enumerate(trainloader):  

				optimizer.zero_grad()
				inputs, labels = data[0], data[1]
				loss1 = lossFunction(self.model(inputs), labels)
				loss2 = lossFunc2(self.model(inputs), labels)
				loss = alpha * loss1 + beta * loss2
				loss1.backward()
				optimizer.step()

			self.model.eval()

			with torch.no_grad():
			
				trainingLoss = lossFunction(self.model(training.inputs), training.labels)
				testingLoss = lossFunction(self.model(testing.inputs), testing.labels)

				self.epochLoss["training"].append(trainingLoss)
				self.epochLoss["testing"].append(testingLoss)
									
				if testingLoss < bestLossLocal:
					bestLossLocal  = testingLoss
					bestModelLocal = deepcopy(self.model)
					#bestEpochLocal = epoch
			
			if getLogLevel() <= 20: # 20 == info
				print("\repoch: %d/%d | loss: %f (%f) %s %s   " %(epoch+1,epochNum, bestLossLocal, testingLoss, loss1.item(), beta*loss2.item()), end = "" if epoch+1 < epochNum else "\n")

		self.model = bestModelLocal



	def _getWrongPredictions(self, maxError = 0.05):

		"""
		Returns a subset of all insufficiently predicted datapoints of our current model

		:param maxError: target relative error by which a prediction is deemed insufficient
		"""

		whichset = self.dataset #self.training

		error = getModelError(self.model, whichset, self.type, returnMean = False)
		subset = []

		for n,e in enumerate(error):

			if e > maxError:
				raw = [i.item() for i in whichset.inputs[n]]
				raw.append(whichset.labels[n].item())
				subset.append(raw)

		subset = Data(subset, self.device)
		return subset


	def _findDelimiter(self, delimiter, predictions, labels):

		"""
		Used for the scipy.optimize.minimize method to converts float output of classification predictions into binary 0,1 values and
		assigns an error value to the given dataset.

		:param delimiter: cutoff value to split float model predictions into 0s and 1s
		:param predictions: predictions of the current model on the validation set
		:param labels: respective labels (true values) corresponding to the predictions

		"""

		right, wrong = 0, 0

		for i in range(len(predictions)):

			if ( labels[i] == 0. and predictions[i] < delimiter ) or ( labels[i] == 1. and predictions[i] > delimiter ):
				right += 1
			else:
				wrong += 1

		return float(wrong)/float(right+wrong)


	def saveLossPlot(self):

		"""
		Plot and save loss progression over training epochs of best performing model.

		"""

		dbPath = self.expres.path
		for i in range(len(dbPath)):
			if dbPath[i:i+8] == 'database':
				dbPath = dbPath[i:]
				break
		path = os.getcwd() + "/" + dbPath + "/performance/"
		Path(path).mkdir(parents=True, exist_ok=True)

		path += str(self.txnameData) + ":" + self.type + "_epochLoss.png"

		trainingLoss = self.winner["epochLoss"]["training"]
		testingLoss = self.winner["epochLoss"]["testing"]

		epo = [n+1 for n in range(len(trainingLoss))]

		title = "epoch loss for %s:%s" % (str(self.txnameData), self.type)

		plt.figure(11)
		plt.title(title, fontsize=20)
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.plot(epo, trainingLoss, label = "training set")
		plt.plot(epo, testingLoss, label = "testing set")
		plt.legend()
		plt.savefig(path)
		plt.close(11)
		logger.info("lossplot saved at %s" % path)

