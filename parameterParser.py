#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: readParameterFile.py
   :synopsis: load parameters for various ML related scripts
.. moduleauthor:: Philipp Neuhuber <ph.neuhuber@gmail.com>

"""

import sys, os, torch
from configparser import ConfigParser


# ------------------------------------------------------------------ #
# main parameter file layout that is referenced in multiple methods  #
# all of these keys will be read from the nn_parameters.ini file and #
# if not found set to 'None' in the main parameter dict              #
# ------------------------------------------------------------------ #

TrainingParameter = {
	"pathing": 	[("databasePath", str), 
				("smodelsPath", str), 
				("utilsPath", str),
				("outputPath", str)],
	"database": [("analysis", str),
				("txName", str),
				("dataselector", str),
				("signalRegion", str),
				("overwrite", str)],
	"dataset": 	[("sampleSize", int), 
				("sampleSplit", float), 
				("externalFile", str),
				("massColumns", int), 
				("refXsecFile", str), 
				("refXsecColumns", int)],
	"computation": [("device", str),
				("cores", int)],
	"validation": [("logFile", bool),
				("lossPlot", bool),
				("runPerformance", bool)],
	"hyparam":	[("optimizer", str), 
				("lossFunction", str), 
				("batchSize", int),
				("activationFunction", str), 
				("epochNum", int), 
				("learnRate", float),
				("layer", int), 
				("nodes", int), 
				("shape", str), 
				("rescaleMethodMasses", str),
				("rescaleMethodTargets", str)]
}



class PermutationDictionary():

	"""
	Custom permutation dictionary that automatically computes all possible combinations of stored lists.
	Individual combinations can be accessed via indexing or iterated through via the 'self.incrIndex' property.
	Suggested use is within a while loop as demonstrated in the 'train.py' method where the current configuration is returned
	through the (very unprofessional) -1 index.

	"""

	def __init__(self, parameterDict):

		self.parameter = parameterDict
		self.combinations = {}
		self.numOfCombinations = 0
		self._index = -1

		paramIndex = {}
		done = False
		firstKey = list(self.parameter.keys())[0]
		lastKey = list(self.parameter.keys())[-1]

		while not done:

			endOfDict = True
			for key in parameterDict:
				currentParamLen = len(self.parameter[key])

				if not key in self.combinations:
					self.combinations[key] = []

					if key != firstKey: paramIndex[key] = 0
					else: paramIndex[key] = -1

				if endOfDict:
					if paramIndex[key] + 1 < currentParamLen:
						paramIndex[key] += 1
						endOfDict = False
					else:
						paramIndex[key] = 0
						endOfDict = True
						done = key == lastKey

			if not done:
				self.numOfCombinations += 1
				for key in self.combinations:
					self.combinations[key].append(paramIndex[key])

	
	@property
	def incrIndex(self):
		self._index += 1
		return self._index < self.numOfCombinations

	@property
	def resetIndex(self):
		self._index = -1

	@property
	def index(self):
		return self._index

	def __len__(self):
		return self.numOfCombinations

	def __getitem__(self, index):

		if isinstance(index, str):
			target = self.parameter[index][self.combinations[index][self._index]]
			if isinstance(target, list): target = target[0]
			return target

		if index == -1: index = self._index
		configuration = {"index": index}
		for key in self.parameter:
			configuration[key] = self.parameter[key][self.combinations[key][index]]
		return configuration

	def __str__(self):
		return str(self.parameter)





class Parameter(dict):

	"""
	Main parameter dictionary that is used throughout the machine learning process. Although individual custom dictionaries may be freely used,
	the aim of this custom dictionary is to manage all necessary parameters more compact and elegantly.

	"""

	def __init__(self, fileName, logLevel):

		"""
		Reads the parameter file and configurates all nested custom (permuation) dictionaries.
		Loads smodels, utils and database and generates a global logger.

		:param fileName: the parameter file that will be read. Defaults to nn_parameters.ini on the same level as this method (string)
		:param logLevel: verbosity level of our logger
		"""

		print("reading %s.." %fileName)

		parser = ConfigParser( inline_comment_prefixes=(';', ) )
		parser.allow_no_value = True
		parser.read(fileName)

		self._parameter = {}

		netTypes = ["regression", "classification"]

		for net in netTypes:
			TrainingParameter[net] = TrainingParameter["hyparam"]
		del(TrainingParameter["hyparam"])

		for key,values in TrainingParameter.items():

			if parser.has_section(key):

				loadedValues = {}

				for line in values:

					keyword = line[0]
					fromat = line[1]

					try:
						param = parser.get(key, keyword).split(",")

						if fromat != str:
							param = [fromat(x) for x in param]
					except: param = [None]

					loadedValues[keyword] = param

			else:
				print("no '{}' section found. Skipping..".format(key)) #logger.info
				loadedValues = None


			self._parameter[key] = loadedValues


		if not self._parameter["database"]["overwrite"][0] in ["always","never","outperforming"]:
			self._parameter["database"]["overwrite"] = ["outperforming"]
			print("invalid overwrite parameter. Allowed options: 'always' 'never' and 'outperforming'. Setting parameter to 'outperforming'") #logger.warning

		self._parameter["database"] = PermutationDictionary(self._parameter["database"])
		hyperParameter = {}
		for net in netTypes:
			hyperParameter[net] = PermutationDictionary(self._parameter[net])
			del(self._parameter[net])
		self._parameter["hyperParameter"] = hyperParameter


		self._parameter["pathing"]["smodelsPath"] = os.path.abspath(self["smodelsPath"])
		self._parameter["pathing"]["utilsPath"] = os.path.abspath(self["utilsPath"])
		self._parameter["pathing"]["databasePath"] = os.path.abspath(self["databasePath"])

		sys.path.append(self["smodelsPath"])
		sys.path.append(self["utilsPath"])
		from smodels.experiment.databaseObj import Database
		self._parameter["smodels-db"] = Database(self["databasePath"])

		import smodels.tools.smodelsLogging as log
		log.setLogLevel(logLevel)
		from smodels.tools.smodelsLogging import logger

		try: device = int(self["device"])
		except: device = self["device"]
		deviceCount = torch.cuda.device_count()
		if isinstance(device, int) and torch.cuda.is_available() and device < deviceCount:
			device = torch.device("cuda:" + str(device))
			logger.info("running on GPU:%d" %device)
		else:
			device = torch.device("cpu")
			logger.info("running on CPU")
		self._parameter["computation"]["device"] = device


	def loadExpres(self):

		"""
		Load current experimental analysis from smodels database

		"""

		analysis 	 = self["database"][-1]["analysis"]
		txName 		 = self["database"][-1]["txName"]
		dataSelector = self["database"][-1]["dataselector"]
		signalRegion = self["database"][-1]["signalRegion"]

		try:
			expres = self["smodels-db"].getExpResults(analysisIDs = analysis, txnames = txName, dataTypes = dataSelector, useSuperseded = True, useNonValidated = True)[0]
			txList = expres.getDataset(signalRegion).txnameList
		except:
			print("No result found for %s: %s [%s] (%s)" % (analysis, txName, signalRegion, dataSelector))
			return False

		for tx in txList:
			if str(tx) == txName:
				break

		self._parameter["expres"] = expres
		self._parameter["txName"] = tx

		return True

		
	def set(self, key, value):
		self._parameter[key] = value

	def __getitem__(self, targetKey):

		"""
		Overload the dictionary indexing method by allowing to bypass sub dictionary notation for easier and less convoluted access to parameters.
		Instead of lets say this[dataset][sampleSize], sampleSize can be directly accessed with this[sampleSize].

		"""

		target = None

		if targetKey in self._parameter:
			target = self._parameter[targetKey]
		else:
			for subdict in self._parameter.values():

				if isinstance(subdict, PermutationDictionary):
					if targetKey in subdict.parameter:
						target = subdict[targetKey]
						break

				if isinstance(subdict, dict) and targetKey in subdict:
					target = subdict[targetKey]
					break
		
		if isinstance(target, list) and len(target) == 1: target = target[0]
		return target


	#def set(self, target, value, subKey = None):
	#def add(self, target, value, subKey = None):

	def __str__(self):
		for key, value in self._parameter.items():
			print(value)



