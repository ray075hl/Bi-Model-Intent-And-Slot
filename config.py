import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


max_length = 60
embedding_size = 256
hidden_size = 256


total_epoch = 10
batch_size = 16
learning_rate = 0.001


