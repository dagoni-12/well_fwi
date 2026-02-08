import torch
import torch.nn as nn
import torch.optim as optim

def f(x):
    return 3 * (torch.sin(5 * x) + 2 * torch.exp(x))

x_true = torch.tensor([0.123, 0.321])
y_true = f(x_true)

x = torch.zeros(2, requires_grad=True)
opt = optim.SGD([x], lr=1e-4)
loss_fn = nn.MSELoss()

for i in range(2000):
    opt.zero_grad()  # Clear gradients from previous iteration
    y = f(x)
    loss = loss_fn(y, y_true)
    loss.backward()  # Compute gradients
    opt.step()  # Update x based on gradients

print(x)
# Expected output (approximately):
# tensor([0.1230, 0.3210], requires_grad=True)