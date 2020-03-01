config = {
	"num_epochs":7,
	"batch_size":128,
	"word_dimension":1000, # The dimensionality of word embeddings
	"image_dimension":512, # input image dimension : 4096 for VGG, 512 resnet
	"model_dimension":1000, # The dimension of the embedding space,
	"learning_rate":0.01,
	"display_freq":50, # how often to display loss : 1 = every batch, 2 = every second batch etc...
	"margin_pairwise_ranking_loss":0.2, # Should be between zero and 1,
	"dataset":"icons_v{}", # visual_fashion_dialog, deepfashion
	"cuda":True, # enable cuda
	"version": 1 # dataset version
}