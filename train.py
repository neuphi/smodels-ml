#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: main.py
   :synopsis: main script to train and validate new neural networks
.. moduleauthor:: Philipp Neuhuber <ph.neuhuber@gmail.com>

"""

import argparse
from parameterParser import Parameter
from mlCore.dataset import DatasetBuilder, Data #set
from mlCore.trainer import ModelTrainer
from mlCore.network import DatabaseNetwork

def main(parameter):

	"""
	Example usage of neural network training methods.
	Reads the parameter file and trains networks for all
	maps it can find.

	:param parameter: Custom parameter dictionary generated from parameterParser.py by reading nn_parameters.ini file. See respective files for more information or parameterParser.py for a full list of keys and values.

	"""

	# ----------------------------------------------------------------------------------- #
	# custom dictionary class that automatically permutates all possible map combinations #
	# ----------------------------------------------------------------------------------- #

	thingsToTrain = parameter["database"]

	# -------------------------------------------------------------------------------- #
	# loop over all analysis map combinations of the parameter file [database] section #
	# -------------------------------------------------------------------------------- #

	while(thingsToTrain.incrIndex):

		# ------------------------------------------------------------------------------------ #
		# load experimental data of current configuration. Stored in key "expres" and "txName" #
		# ------------------------------------------------------------------------------------ #

		if not parameter.loadExpres(): continue

		# -------------------------------------------------------------- #
		# load custom class that will generate our datasets for training #
		# -------------------------------------------------------------- #

		builder = DatasetBuilder(parameter)

		# --------------------------------------------------------------- #
		# optional filter condition for loaded or generated datasets. NYI #			
		# --------------------------------------------------------------- #

		#builder.addFilterCondition(column, condition) eg bigwidths filter, or only every x-th datapoint accepted

		# ---------------------------------------------------------------------------- #
		# train both model types separately, combine them afterwards into one ensemble #
		# ---------------------------------------------------------------------------- #

		winner = {}
		for nettype in ["regression", "classification"]:
			parameter.set("nettype", nettype)


			# ------------------------------------------------- #
			# generate or load dataset used for training 		#
			# output will be custom Dataset class used by torch #
			# ------------------------------------------------- #

			#dataDict = builder.run(nettype, loadFromFile = True)
			
			builder.createDataset(nettype)
			builder.shuffle()

			builder.rescaleMasses()
			builder.rescaleTargets()

			dataset = builder.getDataset()

			# ------------------------------------------------------- #
			# initializing trainer class for current map and net type #
			# ------------------------------------------------------- #

			trainer = ModelTrainer(parameter, dataset)

			# --------------------------------------------------------------------------------------- #
			# running trainer on all hyperparam configurations and saving the best performing network #
			# --------------------------------------------------------------------------------------- #

			winner[nettype] = trainer.run()

			# ---------------- #
			# saving loss plot #
			# ---------------- #

			if parameter["lossPlot"]:
				trainer.saveLossPlot()
		
		# -------------------------------------------------------------------------------------------- #
		# combining best performing regression and classification networks into final ensemble network #
		# -------------------------------------------------------------------------------------------- #

		ensemble = DatabaseNetwork(winner)
		ensemble.save(parameter["expres"], parameter["txName"].txnameData)



if __name__=='__main__':

	ap = argparse.ArgumentParser(description="Trains and finds best performing neural networks for database analyses via hyperparameter search")
	ap.add_argument('-p', '--parfile', 
			help='parameter file', default='nn_parameters.ini')
	ap.add_argument('-l', '--log', 
			help='specifying the level of verbosity (error, warning, info, debug)',
			default = 'info', type = str)
	args = ap.parse_args()

	parameter = Parameter(args.parfile, args.log)

	main(parameter)

