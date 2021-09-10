#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from time import time
from copy import deepcopy
from random import random, shuffle
from scipy.spatial import ConvexHull#, convex_hull_plot_2d
from scipy.spatial import Delaunay
from sklearn.cluster import MeanShift
from mlCore.loadGridPoints import loadInternalGridPoints, loadExternalGridPoints
#from mlCore.auxiliary import loadGridPoints
from torch.utils.data import Dataset # better import?
from sklearn.preprocessing import MinMaxScaler
from smodels.theory.auxiliaryFunctions import rescaleWidth, removeUnits#, unscaleWidth
from smodels.tools import physicsUnits
from smodels.tools.smodelsLogging import logger
from smodels.tools.physicsUnits import GeV, fb, pb


class Data(Dataset):

	"""
	Holds the actual datasets in torch.tensor format, can copy and split itself into subsets

	"""

	def __init__(self, dataset, device):

		"""
		:param dataset: (2dim list:float) or (tuple:list:float) masses and targets to be converted into torch.tensor format
		:param device: (str) 'cpu' or 'gpu:n' n=0,1,... device to run torch on

		"""


		if isinstance(dataset, tuple):
			inputDimension = len(dataset[0][0])
			self.inputs = torch.tensor(dataset[0], dtype=torch.float64).to(device)
			self.labels = torch.tensor(dataset[1], dtype=torch.float64).to(device)
		else:
			inputDimension = len(dataset[0][:-1])
			self.inputs = torch.tensor(dataset, dtype=torch.float64).narrow(1, 0, inputDimension).to(device)
			self.labels = torch.tensor(dataset, dtype=torch.float64).narrow(1, inputDimension, 1).to(device)

		self.inputDimension = inputDimension
		self.device = device


	def __len__(self):
		return self.inputs.size()[0]


	def split(self, sampleSplit):

		"""
		Split dataset into subsets that inherit all parameter such as input dimension or device used.

		:param sampleSplit: (list:float) how to split dataset. sum of elements has to be 1 

		"""

		if sum(sampleSplit) != 1.:
			logger.error("Dataset splice ratios don't add up to 1")

		length = len(self)
		start = 0

		splitData = [[] for i in range(len(sampleSplit))]

		for i in range(len(sampleSplit)):
			if i > 0: start += int(length * sampleSplit[i-1])
			end = int(length * sampleSplit[i])

			splitData[i] = deepcopy(self)
			splitData[i].inputs = self.inputs.narrow(0, start, end)
			splitData[i].labels = self.labels.narrow(0, start, end)

		return splitData
				
	def __getitem__(self, index):
		return (self.inputs[index], self.labels[index])



class DatasetBuilder():

	"""
	The goal of this class is to simplify generating datasets. Possesses numerous methods of data manipulation such as reading and formatting original gridpoints, performing PCA on datapoints, building convex hulls,
	and generating and rescaling datasets. All required meta information is parsed via the parameter dictionary. See parameterParser.py for the content of the parameter dictionary.

	"""

	def __init__(self, parameter):

		"""
		Sets up the dataset generation for a specific map
		:param parameter: Holds all neccessary information to create or load datasets. Can be a simple dict, or the custom class introduced in 'readParameter.py'. If using a custom parameter dictionary make sure to include every key parsed in this init method. (dict:string/float)	

		""" 

		self.smodelsPath = parameter["smodelsPath"]
		self.utilsPath = parameter["utilsPath"]

		self.expres = parameter["expres"]
		self.txnameData = parameter["txName"].txnameData
		self.dataselector = parameter["dataselector"]
		self.signalRegion = parameter["signalRegion"]
		self.full_dim = self.txnameData.full_dimensionality
		self.luminosity = parameter["txName"].globalInfo.getInfo("lumi").asNumber(1/fb)

		self.sampleSize 		  = {"regression": parameter["sampleSize"][0], "classification": parameter["sampleSize"][1]}
		self.sampleSplit 		  = parameter["sampleSplit"]
		self.rescaleMethodMasses  = {"regression": parameter["regression"]["rescaleMethodMasses"], "classification": parameter["classification"]["rescaleMethodMasses"]}
		self.rescaleMethodTargets = {"regression": parameter["regression"]["rescaleMethodTargets"], "classification": parameter["classification"]["rescaleMethodTargets"]}
		self.device 			  = parameter["device"]

		self.refXsecFile = parameter["refXsecFile"]
		if self.refXsecFile != None:
			self.refXsecColumns = parameter["refXsecColumns"]
			self._readRefXsecs()
		else: self.refXsecs = None

		self.externalFile = parameter["externalFile"]

		if self.externalFile != None:
			if os.path.isfile(self.externalFile):
				self.massColumns = parameter["massColumns"]
			else:
				logger.warning("can't find external file (%s) -> will instead try to load grid points from database" %self.externalFile)
				self.externalFile = None
			
			
		logger.info("builder completed for %s" % self.txnameData)




	def _loadGridPoints(self, includeExtremata = False, targetMinimum = 1e-14, rawValues = True):

		"""
		Load grid points from either txnameData orig points or an external file and store them in self._gridPoints and self._gridTargets
		This method is still under construction and subject of constant change, so code is not very pretty at the moment.
		If masses or targets are rescaled - depending on the rescaling method - it is advised to include minima and maxima of each parameter axis in the dataset

		:param includeExtremata: (boolean) (optional) flag to make sure we include minima and maxima of each axis in the dataset
		:param targetMinimum: (float) (optional) target values (effs/ULs) below this threshold will be set to this number instead to avoid having 0s or really small targets in the dataset
		:param rawValues: (boolean) (optional) strip units of masses and values if loaded by internal file

		"""


		if self.externalFile == None:

			logger.info("loading gridpoints from %s.txt" % self.txnameData)
			self._gridPoints, self._gridTargets = loadInternalGridPoints(self.expres, self.txnameData, self.dataselector, self.signalRegion, stripUnits = rawValues)[:-1] #True

		else:

			logger.info("loading gridpoints from %s" % self.externalFile)
			self._gridPoints, self._gridTargets = loadExternalGridPoints(self.externalFile, self.massColumns, includeExtremata, targetMinimum)

		
		if self.refXsecs != None:

			count = 0
			targetMinimum = 1e-14
			for n,point in enumerate(self._gridPoints):

				m0 = point[0]
				xsec = self._getRefXsec(m0)
				fac = self.luminosity * xsec * self._gridTargets[n]

				if fac < 1e-2:
					
					logger.debug("refXsec: %s at %s is below threshhold (%s)" % (point, self._gridTargets[n], fac))
					self._gridTargets[n] = targetMinimum
					count += 1

			logger.info("refXsec: %s/%s gridpoints were set to %s" % (count, len(self._gridTargets), targetMinimum))
					
		else:
			logger.info("no refXsec file specified")


	def _PCA(self, data = None):

		"""
		Perform PCA on any given masspoints (default = original grid points)

		:param data: (2dimlist, float) alternative datapoints, default are original gridpoints of current map

		"""

		if data == None: data = self._gridPoints

		tx = self.txnameData
		ogOrdered = []
		if tx.widthPosition != []:
			for og in data:
				temp, mw = [], []
				for n, m in enumerate(og):	# OG DATA LOADED FROM SMODELS RIGHT NOW DOESNT HAVE EFFS, LOADED FROM FILES HAS THOUGH
					if n == tx.widthPosition[int(n/tx.dimensionality)][1] + 1 + int(n/tx.dimensionality) * tx.dimensionality:
						mw.append(rescaleWidth(m))
					else:
						temp.append(m)
				for w in mw: temp.append(w)
				ogOrdered.append(tx.coordinatesToData(temp))
		else: 
			for og in data:			
				ogOrdered.append(tx.coordinatesToData(og))


		pca = np.array([tx.dataToCoordinates(m, rotMatrix = tx._V, transVector = tx.delta_x) for m in ogOrdered])

		if data == self._gridPoints:
			self._origPCA = pca

		return pca


	def _clusterData(self, bandWidth = 8):

		"""
		Perform meanshift analysis on PCA grid points

		:param bandWidth: (int) (optional) bandwidth of the sklearn.cluster.Meanshift method

		"""

		clustering = MeanShift(bandwidth = bandWidth).fit(self._origPCA)
			
		cluster = [[] for _ in range(len(clustering.cluster_centers_))]
		
		for n, label in enumerate(clustering.labels_):
			cluster[label].append(self._origPCA[n])

		self._origCluster = cluster


	def _addClusterBias(self, sampleSize):
	
		"""
		Adding a bias to draw more points from clusters with non-zero values

		:param sampleSize: (float) inherited from the self._drawRandomPoints method

		"""

		tx = self.txnameData
		clusterMeanVals = []
		zeroClusters, nonZeroClusters = 0, 0
		for n, cluster in enumerate(self._origCluster):

				mean = np.mean(cluster, axis = 0)
				x = tx.coordinatesToData(mean, rotMatrix = tx._V, transVector = tx.delta_x)

				val = tx.getValueFor(x)
				val = removeUnits(val,physicsUnits.standardUnits)

				clusterMeanVals.append(val)

				if val == 0: zeroClusters += len(cluster)
				else: nonZeroClusters += len(cluster)
						

		# If number of gridpoints is greater than sampleSize, 'factor' could be a negative value
		factor = max((sampleSize - zeroClusters) / nonZeroClusters, 1)

		pointsToDraw = []
		for n, cluster in enumerate(self._origCluster):

			if clusterMeanVals[n] > 0:
				pointsToDraw.append(int(factor*len(cluster))+1)
			else:
				pointsToDraw.append(len(cluster))

		return clusterMeanVals, pointsToDraw

	

	def _getHullPoints(self):

		"""
		Algorithm that finds all hull edges of original gridpoints. The reason we use this in addition 
		to the scipy.spatial.ConvexHull method is that CH only identifies vertices.
		Used to generate convex hull dataset for the classification network

		"""

		massesSorted = [sorted(self._gridPoints,key=lambda l:l[n]) for n in range(len(self._gridPoints[0]))]
		massesHull = [[] for _ in range(len(massesSorted))]

		for k in range(len(massesSorted)):

			massesSortedReduced = []
			lastMass = 0.
			totalMasses = len(massesSorted[k])

			for n in range(totalMasses):
				subset = []
				currentMass = massesSorted[k][n][k]
				if lastMass == currentMass:
					continue
				lastMass = currentMass
				for i in range(n, totalMasses):
					if massesSorted[k][i][k] == currentMass:
						subset.append(massesSorted[k][i])
					else: break

				massesSortedReduced.append(np.max(subset, axis=0).tolist())

			massesHull[k] = massesSortedReduced


		# getting rid of duplicate axes

		n = 0
		while n < len(massesHull)-1:
			n += 1
			for m in massesHull[:n]:
				if massesHull[n] == m:
					massesHull.pop(n)
					n -= 1
					continue


		return massesHull

	
	def _readRefXsecs(self):

		"""
		Load reference cross sections for the current analysis.

		"""

		from slha.addRefXSecs import getXSecsFrom

		fiLe = self.refXsecFile
		col  = self.refXsecColumns

		if col != None:
			xsecs = getXSecsFrom(fiLe, columns={"mass":col[0],"xsec":col[1]})
		else:
			xsecs = getXSecsFrom(fiLe)
			
		dic = {"masses":[],"xsecs":[]}

		for key,value in xsecs.items():
			dic["masses"].append(key)
			dic["xsecs"].append(value*1e3) # pb -> fb

		self.refXsecs = dic
	

	def _getRefXsec(self, mother):

		"""
		Returns an approximation of the reference cross section for a given mass point and topology.
		The respective xsec file is read in self._readRefXsecs. 

		:param mother: (float) the mother particle of a given mass point

		"""

		for n, mass in enumerate(self.refXsecs["masses"]):
			if mass > mother:
				return self.refXsecs["xsecs"][n]

		return self.refXsecs["xsecs"][-1]

		


	def _isOnHull(self, point):

		"""
		Returns boolean if parsed mass point is within the convex hull of original grid points

		:param point: (float) single mass point array

		"""

		mp = [point[n] for n in self._delauneyAxes]

		if self._delauney.find_simplex(mp)>=0:
			return 1.
		
		return 0.


	def _getHullDelauney(self):

		"""
		Calculates convex hull of self._gridPoints, weeds out any non vertex points
		and generates delauney triangulation so we can build a classification dataset without smodels delauney
		The first part gets rid of duplicate axes which would cause problems with scipy.spatial.ConvexHull

		"""

		if self.externalFile != None:
			axes, dupl = [], []
			for n,c in enumerate(self.massColumns[:-1]):
				if not c in dupl:
					dupl.append(c)
					axes.append(n)
		else:
			axes, dupl = [], []
			for n, m in enumerate(self._gridPoints[0]):
				if not m in dupl:
					dupl.append(m)
					axes.append(n)


		self._delauneyAxes = axes

		splicedData = np.array([np.array(self._gridPoints)[:, a] for a in axes]).T
		hull = ConvexHull(splicedData)

		hullPoints = []
		for vert in hull.vertices:
			hullPoints.append(splicedData[vert])

		self._delauney = Delaunay(hullPoints)


	def _drawRandomPointsRegression(self):

		"""
		Draws random sample points for regression dataset
		1. add bias to each cluster to draw more points from high density regions
		2. draw points for each cluster

		"""

		particles = [0 for _ in range(self.full_dim)]
		tx = self.txnameData

		#rescaleInputs = tx.widthPosition != []

		drawnMasses = []
		drawnTargets = []

		clusterMeanVals, pointsToDrawPerCluster = self._addClusterBias(self.sampleSize["regression"])
			
		zeroes = 0
		totalPointsDrawn = 0 #sum(pointsToDrawPerCluster)

		for n, cluster in enumerate(self._origCluster):

			mean = np.mean(cluster, axis = 0)
			std = np.std(cluster, axis = 0)
			logger.debug("cluster %s/%s" % (n+1, len(self._origCluster)))
			pointsLeft = pointsToDrawPerCluster[n]

			while pointsLeft > 0:

				rand = []
				for i in range(tx.dimensionality):
					rand.append(np.random.normal(mean[i], 50. + 4.*std[i]))

				x = tx.coordinatesToData(rand, rotMatrix = tx._V, transVector = tx.delta_x)
				val = tx.getValueFor(x)
				val = removeUnits(val,physicsUnits.standardUnits)

				if self.refXsecs != None and val != None:
					x0 = x
					while type(x0) == list or type(x0) == tuple: x0 = x0[0]
					x0 = x0.asNumber(GeV)
					xsec = self._getRefXsec(x0)

					thresh = self.luminosity * xsec * val

					if thresh < 1e-2:
						val = 0.

				if type(val) != type(None) and ( clusterMeanVals[n] == 0 or (val != 0. or random() < 0.15) ): #0.1
						
					pointsLeft -= 1
					totalPointsDrawn += 1
					print("points drawn: %s" % totalPointsDrawn)
						
					if val == 0.:
						zeroes += 1

					strippedUnits = tx.dataToCoordinates(x)
					drawnMasses.append(strippedUnits)
					drawnTargets.append(val)

				if totalPointsDrawn >= 1000: break

		logger.debug("%s%% are zero." % round(100.*(zeroes/len(drawnMasses)), 3))

		self.masses = np.array(drawnMasses)
		self.targets = np.array(drawnTargets)


	def _drawRandomPointsClassification(self):

		"""
		Draws random sample points for classification dataset
		1. get hull points
		2. delauney triangulation for hull vertices only
		3. PCA on hull points
		4. draw points around hull edges
		5. fill rest of dataset with random points of grid point clusters

		"""

		samplesLeft = self.sampleSize["classification"]

		particles = [0 for _ in range(self.full_dim)]
		tx = self.txnameData

		width = tx.widthPosition
		#rescaleInputs = width != []

		drawnMasses = []
		drawnTargets = []

		hullPoints = self._getHullPoints()
		self._getHullDelauney()

		hP_PCA = []
		numOfHullPoints = 0
		for axisPoints in hullPoints:
			numOfHullPoints += len(axisPoints)

			temp = self._PCA(axisPoints)
			hP_PCA.append(temp)

		samplesPerHullPoint = max(1,int(( self.sampleSize["classification"] / numOfHullPoints ) * 0.95)) #0.15)


		for currentMassHull in hP_PCA:

			#mean = np.mean(currentMassHull, axis = 0)
			std = np.std(currentMassHull, axis = 0)

			for point in currentMassHull:

				samplesPerHullPointLeft = samplesPerHullPoint

				while(samplesPerHullPointLeft > 0):

					rand = []
					for i in range(tx.dimensionality):
						rand.append(np.random.normal(point[i], 0.15*std[i]))

					masses = tx.coordinatesToData(rand, rotMatrix = tx._V, transVector = tx.delta_x)
					masses = tx.dataToCoordinates(masses)

					if all([m >= 0 for m in masses]):

						val = self._isOnHull(masses)

						drawnMasses.append(masses)
						drawnTargets.append(val)

						samplesLeft -= 1
						samplesPerHullPointLeft -= 1

						logger.debug("points drawn: %s" % len(drawnMasses))

			
		samplesLeft = 0
		clusterMeanVals, pointsToDrawPerCluster = self._addClusterBias(samplesLeft)

		for n, cluster in enumerate(self._origCluster):

			mean = np.mean(cluster, axis = 0)
			std = np.std(cluster, axis = 0)
			std = [max(s,10) for s in std]

			pointsLeft = pointsToDrawPerCluster[n]

			while pointsLeft > 0:

				rand = []
				for i in range(tx.dimensionality):
					rand.append(np.random.normal(mean[i], 2.5*std[i]))

				masses = tx.coordinatesToData(rand, rotMatrix = tx._V, transVector = tx.delta_x)
				masses = tx.dataToCoordinates(masses)

				if all([m >= 0 for m in masses]):

					val = self._isOnHull(masses)

					drawnMasses.append(masses)
					drawnTargets.append(val)
							
					pointsLeft -= 1
			

		self.masses = np.array(drawnMasses)
		self.targets = np.array(drawnTargets)
	


	def _drawRandomPoints(self):

		"""
		Generates datasets for training and evaluation and returns them as custom 'Data' class
		1. reads original grid points and PCA's them to reduce dimensionality
		2. mean-shift clusters PCA data to get points of high information density
		3. draw random points

		"""

		if not hasattr(self, "_origPCA"):
			self._PCA()

		if not hasattr(self, "_origCluster"):
			self._clusterData()

		t0 = time()
		logger.info("generating dataset.. ")

		if self.nettype == "regression":
			self._drawRandomPointsRegression()

		else:
			self._drawRandomPointsClassification()

		logger.info("dataset generation completed (%ss)" % (round(time() - t0, 3)))


	def createDataset(self, nettype):

		"""
		Generate self.masses and self.targets for the dataset. Points will either be drawn randomly from the txnamedata interpolation
		or read from an external file.

		:param nettype: (string) 'regression' or 'classification' type dataset will be generated.

		"""

		self.nettype = nettype

		if not hasattr(self, "_gridPoints"):
			self._loadGridPoints()	


		if self.nettype == "regression":

			if self.externalFile != None:
				self.masses = np.array(self._gridPoints)
				self.targets = np.array(self._gridTargets)

			else:
				self._drawRandomPoints()


		elif self.nettype == "classification":
			self._drawRandomPoints()

			#self.masses = np.array(self._gridPoints)
			#self.targets = np.array(self._gridTargets)


	def rescaleMasses(self, method = None):

		"""
		Rescale masses either via minmax scaler or standard score.

		:param method: (optional) (string) Specify which method to use. Currently available: 'minmaxScaler' and 'standardScore'. You can override the stored rescale method with 'null' argument

		"""

		if method == None:
			method = self.rescaleMethodMasses[self.nettype]

		if method == "minmaxScaler":
			scaler = MinMaxScaler(feature_range=(1, 100))
			scaler = scaler.fit(self.masses)
			self.masses = scaler.transform(self.masses)

			if not "rescale" in self.__dict__:
				self.rescale = {}

			self.rescale["masses"] = {"method": method, "scaler": scaler}

		elif method == "standardScore":

			mean = np.mean(self.masses, axis = 0)
			std = np.std(self.masses, axis = 0)
			self.masses = (self.masses - mean) / std

			if not "rescale" in self.__dict__:
				self.rescale = {}

			self.rescale["masses"] = {"method": method, "mean": mean, "std": std}

		elif method == "null":
			logger.info("masses will not be rescaled")

		else:
			logger.error("%s: unrecognized rescale method." % method)



	def rescaleTargets(self, method = None, lmbda = None):

		"""
		Rescale targets via boxcox method or log. Mainly used for LLP maps with very low efficiencies.

		:param method: (optional) (string) Overwrite rescaling method. Options: 'log', 'boxcox'.
		:param lmbda: (optional) (float) Used for a fixed boxcox transformation. If set to 'None' boxcox will find an optimal lmbda automatically. Default = None

		"""

		if method == None:
			method = self.rescaleMethodTargets[self.nettype]

		if method == "boxcox":

			from scipy.stats import boxcox

			if lmbda == None:
				self.targets, lmbda = boxcox(self.targets)
			else:
				self.targets = boxcox(self.targets, lmbda)

			logger.info("lambda: %f" % lmbda)

			if not "rescale" in self.__dict__:
				self.rescale = {}

			self.rescale["targets"] = {"method": method, "lambda": lmbda}
			#self.targets = -self.targets

		elif method == "log":

			if not "rescale" in self.__dict__:
				self.rescale = {}

			self.targets = np.log10(self.targets)

			self.rescale["targets"] = {"method": method}

		elif method == "standardScore":

			mean = np.mean(self.targets, axis = 0)
			std = np.std(self.targets, axis = 0)
			self.targets = (self.targets - mean) / std

			if not "rescale" in self.__dict__:
				self.rescale = {}

			self.rescale["targets"] = {"method": method, "mean": mean, "std": std}


		elif method == "null":
			logger.info("targets will not be rescaled")


		self.targets = np.array(self.targets)[np.newaxis]
		self.targets = self.targets.T


	def shuffle(self):

		"""
		Simply shuffle current masses and targets.

		"""

		indices = np.arange(self.targets.shape[0])
		np.random.shuffle(indices)

		self.masses = self.masses[indices]
		self.targets = self.targets[indices]


	def getDataset(self, fullSet = True, splitSet = True, rescaleParams = True):

		"""
		Generate output dictionary.

		:param fullSet: (boolean) Return the full dataset.
		:param splitSet: (optional) (boolean) Split and return full set into subsets specified via self.sampleSplit list.
		:param rescaleParams: (optional) (boolean) Return used rescale parameters of 'self.rescaleMasses' and 'self.rescaleTargets' method.

		:return output: (dict) May contain keys 'full', 'training', 'testing', 'validation' and 'rescaleParams'.
		"""

		output = {}

		if rescaleParams:
			if hasattr(self, "rescale"):
				output["rescaleParams"] = self.rescale

				keys = ["masses", "targets"]
				for key in keys:
					if not key in output["rescaleParams"]:
						output["rescaleParams"][key] = {"method": None}

			else:
				output["rescaleParams"] = {"masses": {"method": None}, "targets": {"method": None}}

		if fullSet or splitSet:
			full = Data((self.masses, self.targets), self.device)

		if fullSet:
			output["full"] = full

		if splitSet:
			splitset = full.split(self.sampleSplit)
			output["training"]   = splitset[0]
			output["testing"]    = splitset[1]
			output["validation"] = splitset[2]

		return output
	
