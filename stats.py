import numpy as np
import torch 
from torch import nn

expected_output = torch.from_numpy(np.load("output.npy").reshape(-1))
he_output = torch.from_numpy(np.load("thirdparty/he-man-concrete/demo/result.npy").reshape(-1))
concrete_output = torch.from_numpy(np.load("output_concrete.npy").reshape(-1))

softmax = nn.Softmax(dim=0)
print("\nLogits from the original pytorch model: ")
expected_prob = softmax(expected_output)
print(expected_prob)
print("\nLogits from the HE-transformed model: ")
he_prob = softmax(he_output)
print(he_prob)
print("\nLogits from the concrete model: ")
concrete_prob = softmax(concrete_output)
print(concrete_prob)

loss = nn.CrossEntropyLoss()
print("\nCross Entropy loss for the HE-transformed model: ")
print(loss(expected_prob, he_prob))
print("\nCross Entropy loss for the concrete model: ")
print(loss(expected_prob, concrete_prob))
