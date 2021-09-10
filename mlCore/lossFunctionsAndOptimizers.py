import os, torch
from torch import nn
import numpy as np

def RMSE(predicted, label, reduction = "mean", denomOffset = 0.): #1e-4

	"""
	Custom Relative Mean Squared Error loss function for network training and validation

	:param predicted: Model prediction (torch tensor)
	:param label: Target values (torch tensor)
	:param reduction: Specify if output tensor will be squared rel error with length of inputs or a single mean square root value over all inputs (string) (optional)
	:param denomOffset: Denominator offset, useful for training with zero labels to avoid divergence (float) (optional)

	"""

	loss = ((predicted-label)/(label+denomOffset))**2
	if reduction == "mean": loss = torch.sqrt(torch.mean(loss))
	return loss

def loadOptimizer(optimizerName, model, learnRate):

	"""
	Translates the parameter files optimizer string into torch optimizers and loads them into the trainer class. Can be expanded at will.

	:param optimizerName: Optimizer to be loaded (string)
	:param model: Current model that is to be trained (torch.nn.Module child)
	:param learnRate: Applied learn rate (float)

	"""

	if optimizerName == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)
		logger.warning("Invalid optimizer selected. Only Adam is supported currently. Continuing on Adam")

	return optimizer


def loadLossFunction(lossFunctionName, device):

	"""
	Translates the parameter files loss function string into loss function methods. Can be expanded at will.

	:param lossFunctionName: Lossfunction to be loaded (string)
	:param device: Sends the lossfunction either to CPU or any single GPU training is being performed on (string)
	"""

	if lossFunctionName == "MSE": lossFunction = nn.MSELoss(reduction = 'mean').to(device)
	elif lossFunctionName == "RMSE": lossFunction = RMSE
	elif lossFunctionName == "BCE": lossFunction = nn.BCELoss(reduction = 'mean').to(device)

	return lossFunction


def getModelError(model, dataset, nettype, returnMean = True):

	"""
	Quick hack to gain model performance over specific dataset.
	Useful to have a uniform error check regardless of loss function used
	during training.

	:param model: Model that needs to be evaluated (torch.nn.Module child)
	:param dataset: Data to test the model with (torch.utils.data.Dataset child)
	:param nettype: Either "regression" or "classification" depending on the model architecture used (string)
	:param returnMean: Output will be either a loss tensor with input length for each datapoint or the mean loss over all inputs (boolean) (optional)

	"""

	predictions = model(dataset.inputs)
	labels = dataset.labels
	
	error = []

	if nettype == "regression":

		for n, label in enumerate(labels):

			l = label.item()
			p = predictions[n].item()

			if l > 0:
				e = np.sqrt((( p - l ) / l)**2)
			else:
				e = p


			error.append(e)

	else:

		modelState = model.training
		model.training = True

		for n, label in  enumerate(labels):

			l = label.item()
			p = predictions[n].item()

			if l == p: e = 1.
			else: e = 0.

			error.append(e)

		model.training = modelState

	error = np.array(error)

	if returnMean:
		mean = np.mean(error)
		std = np.std(error)
		return mean, std
	
	return error
