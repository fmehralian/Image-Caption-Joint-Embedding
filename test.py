from data import Data
from iconClasses.model import Model

def main():

	data = Data()	
	data.load_dictionaries() # very important - load dictionaries 

	model = Model(data)
	model.input_name = "best" # specify save model name (best)
	model.load() # load the saved model weights

	model.evaluate(data) # evaluate the data

	return

if __name__ == '__main__':
	main()
	print("\nScript done :)")