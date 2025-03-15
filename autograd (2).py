import torch
import torch.nn as nn
import numpy as np
# x = torch.randn(3,requires_grad=True)
# print(x)

# # y = x+2
# # print(y)

# # z=y*y*2
# # z = z.mean()
# # print(z)

# # z.backward()
# # print(x.grad)



# dummy model

# weights = torch.ones(4,requires_grad=True)

# for epoch in range(100):
#     model_output = (weights*3).sum()
    
#     model_output.backward() 
#     print(weights.grad)
#     weights.grad.zero_()

# optimizer = torch.optim.SGD(weights,lr=0.1)
# optimizer.step()
# optimizer.zero_grad()

X = np.array([1,2,3,4],dtype= np.float32)
Y = np.array([2,4,6,8],dtype= np.float32)

w = 0.0

def forward(x):
    return w*x


def gradient(x,y,y_predicted):
    return np.dot(2+x,y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3}')

l_r = 0.01
n_iters = 10

loss = nn.MSELoss()
for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y,y_pred)
    
    dw = gradient(X,Y,y_pred)

    w -= l_r *dw

    if epoch %1 ==0:
        print(f'{epoch+1}:w = {w:.3f},loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')