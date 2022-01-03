import torch

# f = w * x
x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8, 10, 12], dtype=torch.float32)

# init weight
# 我們希望 Pytorch 幫我們計算更新的 Gradient 變數是 w，所以一定要對這個變數開啟 requires_grad
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x
    
# set up loss function as mean square error
def loss(y, y_predicted):
    return ((y-y_predicted) ** 2).mean()

# Training
learning_rate = 0.01
n_iters = 30

for epoch in range(n_iters):
    # perdiction = forward pass
    y_pred = forward(x)

    # loss
    l = loss(y, y_pred)

    # gradient descent is where calculate gradient and update parameters
    # so gradient descent here includes gradients and update weights
    # 原本在 Python 的 example 還需要自己建立 Gradient 函式
    # gradients = backward pass
    l.backward() # calculate dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')
    
    # zero gradients，要記得歸零每次運算的 gradients，否則會累加
    w.grad.zero_()

print(f'Prediction after training: f(5) = {forward(5): .3f}')