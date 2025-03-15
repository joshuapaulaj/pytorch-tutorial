#tensor
import torch


x  = torch.ones(2,2,dtype=torch.int)
print(x)
print(x.dtype)
print(x.size())

a = torch.rand(2,2)
b = torch.rand(2,2)
b.add_(x)
print(b)