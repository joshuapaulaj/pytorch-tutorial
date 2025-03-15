import torch 
import numpy as np
# x = torch.zeros(2,3,dtype=torch.float16)
# print(x.dtype)

# x = torch.tensor([2.6,0.1])
# print(x.size())


# x = torch.rand(2,2)
# y = torch.rand(2,2)
# # print(x)
# # print(y)
# # z = x+y
# # print(z)

# y.add_(x)
# print(y)

a = np.ones(5)
print(a)

b = torch.from_numpy(a)
print(b)

a += 1 
print(a)
print(b)