import torch

# x = torch.empty(2,2,2,3)
# print(x)

# x = torch.rand(2,2)
# print(x)

# x = torch.ones(2,2,dtype = torch.float16)
# print(x.size())

# x = torch.tensor([2.5,0.1])
# print(x)

# y.mul_(x)

# import torch
# import numpy

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(type(b))

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5,device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x + y
#     z = z.to("cpu")
#     print(z)
# else:
#     print("no")