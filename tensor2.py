import torch
import numpy as np

a = torch.ones(5)
b = a.numpy()
print(type(b))