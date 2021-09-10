


smodelsPath = ""
databasePath = "../../smodels-database"
from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb, pb
from mlCore.network import DatabaseNetwork

import numpy as np

def getPrediction(expres, txnameData, massPoint):

	massPoint = [txnameData.dataToCoordinates(massPoint)]
	model = DatabaseNetwork.load(expres, txnameData)
	return model(massPoint)


	


if __name__ == "__main__":

	analysis 	 = "ATLAS-SUSY-2016-32"
	txName 		 = "THSCPM1b"
	dataselector = "efficiencyMap"
	signalRegion = "SR1FULL_175"
	massPoint = [[(500.*GeV, 0)], [(500.*GeV, 0)]]

	db = Database(databasePath)

	expres = db.getExpResults(analysisIDs = analysis, txnames = txName, dataTypes = dataselector, useSuperseded = True, useNonValidated = True)[0]
	txList = expres.getDataset(signalRegion).txnameList #"SR1FULL_175"

	for tx in txList:
		if str(tx) == txName:
			txnameData = tx.txnameData
			break


	prediction = getPrediction(expres, txnameData, massPoint)
	print("Efficiency predicted for %s: %f" % (massPoint, round(prediction[0], 5)))
