import os
import numpy as np
from smodels.tools.stringTools import concatenateLines
from smodels.tools.physicsUnits import GeV, fb, pb
from smodels.theory.auxiliaryFunctions import rescaleWidth
from smodels.tools.smodelsLogging import logger


def loadInternalGridPoints(expres, txnameData, dataselector, signalRegion, singleLines = True, stripUnits = True):

	"""
	Quick hack to load the original grid points within the database txname.txt files for a given analysis.

	:param expres: SModelS experimental result to be loaded (smodels.ExpResult)
	:param txnameData: SModelS topology to be loaded (smodels.txnameData)
	:param dataselector: Either "upperLimit" or "efficiencyMap" (string)
	:param signalRegion: For efficiency maps select a signal region, for upperlimit maps select "None" (float/None)
	:param singleLines: Return gridpoints as either single arrays for each mass point, or nested arrays for each decay branch (boolean) (optional)
	:param stripUnits: Output will be either floats or floats*smodels.tools.physicsUnits (boolean) (optional)
	
	"""

	if dataselector == "upperLimit":
		whichTag = "upperLimits"
	else:
		whichTag = "efficiencyMap"

	for tx in expres.getTxNames():
		if tx.txnameData == txnameData:
			tx = str(tx)
			break


	if dataselector == "upperLimit":
		filePath = expres.path + '/data/' + tx + '.txt'
	else:
		filePath = expres.path + '/' + signalRegion + '/' + tx + '.txt'

	with open(filePath) as txtFile:
		txdata = txtFile.read()
	content = concatenateLines(txdata.split("\n"))
	tags = [line.split(":", 1)[0].strip() for line in content]

	for i,tag in enumerate(tags):
		if not tag: continue
		line = content[i]
		value = line.split(':',1)[1].strip()
		if ";" in value: value = value.split(";")

		if tag == whichTag:
			data = value
			break


	origData, values = [], []

	data = data.split("],[[[")

	if data[0] != data[0].replace('*fb',''): unitValues = fb
	elif data[0] != data[0].replace('*pb',''): unitValues = pb
	else: unitValues = 1.

	if stripUnits: units = [1., 1.]
	else: units = [GeV, unitValues]
	
	for line in data:
		line = line.replace('[[[[','')
		line = line.replace('],[',',')
		line = line.replace(']]','')
		line = line.replace('*GeV','')
		line = line.replace('(', '')
		line = line.replace(')', '')
		
		#if line != line.replace('*fb',''): UNITS = "*fb"
		#elif line != line.replace('*pb',''): UNITS = "*pb"
		#else: UNITS = None
			
		line = line.replace('*fb','')
		line = line.replace('*pb','')

		point = line.split(",")
		masses = [float(p)*units[0] for p in point[:-1]]
		dHalf = int(0.5*len(masses))

		if not singleLines:
			masses = [[m for m in masses[0:dHalf]],[m for m in masses[dHalf:]]]

		origData.append(masses)

		value = float(point[-1])*units[1]
		values.append(value)

	return origData, values, units



def loadExternalGridPoints(externalFile, massColumns, includeExtremata = False, targetMinimum = 1e-14):

	"""
	Load grid points from either txnameData orig points or an external file and store them in self._gridPoints and self._gridTargets
	This method is still under construction and subject of constant change, so code is not very pretty at the moment.
	If masses or targets are rescaled - depending on the rescaling method - it is advised to include minima and maxima of each parameter axis in the dataset

	:param externalFile: (string) file to load gridpoints from
	:param massColumns: (list, int) how to interpret contents of 'externalFile'. Example: [0,0,-2,-2,7] for THSCPM1b: column 0 corresponds to m0, column 2 to widths (widths are identified by being negative), column 7 holds all target values
	:param includeExtremata: (boolean) (optional) flag to make sure we include minima and maxima of each axis in the dataset. (Important for some rescaling methods to ensure consistency)
	:param targetMinimum: (float) (optional) target values (effs/ULs) below this threshold will be set to this number instead to avoid having 0s or really small targets in the dataset

	"""
		
	with open(externalFile) as txtFile:
		raw = txtFile.read()

	lines = raw.split("\n")[1:]

	dataset_masses = []
	dataset_targets = []
	count = [0 for n in range(20)]
	squished = 0
	totalLen = len(lines) - 1

	dimensionality = len(massColumns) - 1
	minimum = [[[1e4 for _ in range(dimensionality)], [0]] for _ in range(dimensionality)]
	maximum = [[[0. for _ in range(dimensionality)], [0]] for _ in range(dimensionality)]

	for line in lines[:-1]:
		values = line.split()
		masses = []

		width = 0

		for x in massColumns[:-1]:
			if x < 0:
				x = -x
					
				width = float(values[x])
				masses.append(rescaleWidth(float(values[x])))
			else:
				masses.append(float(values[x]))

		target = max(targetMinimum, float(values[massColumns[-1]]))
			
		"""
		if self.refXsecs != None:
			m0 = masses[0]
			xsec = self._getRefXsec(m0)

			if self.luminosity * xsec * target < 1e-2:
					
				if target > 0.: squished += 1
				target = 0.
		"""

		for n,mass in enumerate(masses):
			if minimum[n][0][n] > mass:
				minimum[n] = [masses, target]
			elif maximum[n][0][n] < mass:
				maximum[n] = [masses, target]
					

		#if target > 1e-14:# and m0 > 250. and width < 1e-17:
		#if target > targetMinimum:# or random() > 0.9:

		###### testing purposes ######
		if True: #target != targetMinimum:
			dataset_masses.append(masses)
			dataset_targets.append(target)
		###### ---------------- ######

		if target == 0.: x = 0
		else:
			x = int(-np.log10(target)) + 1
		count[x] += 1

		
	if includeExtremata:

		for mini in minimum:
			if not mini[0] in dataset_masses:
				target = max(targetMinimum,mini[1])
				print("t:", target)
				dataset_masses.append(mini[0])
				dataset_targets.append(target)
		for maxi in maximum:
			if not maxi[0] in dataset_masses:
				target = max(targetMinimum,maxi[1])
				dataset_masses.append(maxi[0])
				dataset_targets.append(target)
		
	for n,c in enumerate(count[:10]):
		if n == 0: p = 0
		else: p = 10**-n
		logger.debug("# effs at ~%s: %s" % (p, c))

		

	logger.debug("length of dataset: %s" %len(dataset_masses))
	logger.debug("number of eff squished to 0: %s (%s%%)" % (squished, 100*round(squished/totalLen,2)))
	logger.debug("percentage of 0 efficiencies: %s%%" %(round(100*count[0]/totalLen, 2)))
	logger.debug("taget minimum: %s" % min(dataset_targets))
	logger.debug("taget maximum: %s" % max(dataset_targets))

	return dataset_masses, dataset_targets
