from data import Data
from settings import config
from iconClasses.model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer

# Load data
data = Data()

# track score to save best model
score = 0

# Use K fold cross validation for model selection
for train, test, fold in data.k_folds(10):

	# Prepare data to use the current fold
	data.process(train, test, fold)

	# Load model
	model = Model(data)

	# Model loss function
	loss = Loss()

	# Optimizer
	optimizer = Optimizer(model)

	# Begin epochs
	for epoch in range(config["num_epochs"]):

		# Process batches
		for caption, image_feature, contents in data:
			pass

			# Pass data through model
			caption, image_feature = model(caption, image_feature)

			# Compute loss
			cost = loss(caption, image_feature)

			# Zero gradient, Optimize loss, and perform back-propagation
			optimizer.backprop(cost)

		# Evaluate model results
		model.evaluate(data)

	# Final evaluation - save if results are better
	model_score = model.evaluate(data)
	if model_score > score:
		score = model_score
		model.save()
		data.save_dictionaries()
