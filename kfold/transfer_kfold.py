from data import Data
from settings import config
from iconClasses.model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer

if __name__ == "__main__":

	# Load data
	data = Data()
	data.load_dictionaries()

	# track score to save best model
	score = 0

	# Use K fold cross validation for model selection
	for train, test, fold in data.k_folds(5):		

		# Prepare data to use the current fold
		data.process(train, test, fold, create_dictionaries=False)

		# Load model
		model = Model(data)
		model.input_name = "best"
		model.output_name = "final"
		model.load()

		# Model loss function
		loss = Loss()
	
		# Optimizer 
		optimizer = Optimizer(model)

		# Begin epochs
		for epoch in range(config["num_epochs"]):
			print("[EPOCH]", epoch+1)

			# Process batches
			for caption, image_feature in data:
				pass			

				# Pass data through model
				caption, image_feature = model(caption, image_feature)

				# Compute loss
				cost = loss(caption, image_feature)			

				# Zero gradient, Optimize loss, and perform back-propagation
				optimizer.backprop(cost)

			# Evaluate final model results					
			model.evaluate(data)

		# Final evaluation - save if results are better		
		print("\nFinal evaluation:")
		model_score = model.evaluate(data)
		if model_score > score:
			score = model_score
			model.save()			
			print("[BEST SCORE]", score)

		# TODO - print k-fold validation results, averaged across all models

	print("\n[SCRIPT] complete")