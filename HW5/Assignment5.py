import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def chain_rule():
    """
    Compute df/dz, df/dq, df/dx, and df/dy for f(x,y,z)=xy+z,
    where q=xy, at x=-2, y=5, z=-4.
    Return them in this order: df/dz, df/dq, df/dx, df/dy. 
    """ 
    x = torch.tensor(-2.0, requires_grad=True)
    y = torch.tensor(5.0, requires_grad=True)
    z = torch.tensor(-4.0, requires_grad=True)
    q = x * y
    q.retain_grad()
    f = q + z
    
    f.backward()

    df_dz = z.grad.item()  
    df_dq = q.grad.item() 
    df_dx = x.grad.item()
    df_dy = y.grad.item()  

    return df_dz, df_dq, df_dx, df_dy

def ReLU():
    """
    Compute dx and dw, and return them in order.
    Forward:
        y = ReLU(w0 * x0 + w1 * x1 + w2)

    Returns:
        dx -- gradient with respect to input x, as a vector [dx0, dx1]
        dw -- gradient with respect to weights (including the third term w2), 
              as a vector [dw0, dw1, dw2]
    """
    x0 = torch.tensor(-1.0, requires_grad=True)
    x1 = torch.tensor(-2.0, requires_grad=True)
    w0 = torch.tensor(2.0, requires_grad=True)
    w1 = torch.tensor(-3.0, requires_grad=True)
    w2 = torch.tensor(-3.0, requires_grad=True)

    y = F.relu(w0 * x0 + w1 * x1 + w2)

    y.backward()

    dx = [x0.grad.item(), x1.grad.item()]
    dw = [w0.grad.item(), w1.grad.item(), w2.grad.item()]

    return dx, dw

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are 
    a=0.37, b=1.37, and c=0.73.  
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    x0 = torch.tensor(-1.0, requires_grad=True)
    x1 = torch.tensor(-2.0, requires_grad=True)
    w0 = torch.tensor(2.0, requires_grad=True)
    w1 = torch.tensor(-3.0, requires_grad=True)
    w2 = torch.tensor(-3.0, requires_grad=True)

    a = torch.exp(-(w0 * x0 + w1 * x1 + w2))
    b = 1 + a
    c = 1 / b

    a = round(a.item(), 4)
    b = round(b.item(), 4)
    c = round(c.item(), 4)

    return a, b, c

def chain_rule_b():
    """
    In the lecture notes, the backward pass values are
    ±0.20, ±0.39, -0.59, and -0.53.  
    Calculate these numbers to 4 decimal digits 
    and return in order of gradients for w0, x0, w1, x1, w2.
    """
    x0 = torch.tensor(-1.0, requires_grad=True)
    x1 = torch.tensor(-2.0, requires_grad=True)
    w0 = torch.tensor(2.0, requires_grad=True)
    w1 = torch.tensor(-3.0, requires_grad=True)
    w2 = torch.tensor(-3.0, requires_grad=True)

    f = 1 / (1 + torch.exp(-(w0 * x0 + w1 * x1 + w2)))
    f.backward()

    gw0 = torch.round(w0.grad, decimals=4)
    gx0 = torch.round(x0.grad, decimals=4)
    gw1 = torch.round(w1.grad, decimals=4)
    gx1 = torch.round(x1.grad, decimals=4)
    gw2 = torch.round(w2.grad, decimals=4)

    return gw0, gx0, gw1, gx1, gw2

def backprop_a():
    """
    Let f(w,x) = torch.tanh(w0x0+w1x1+w2).  
    Assume the weight vector is w = [w0=5, w1=2], 
    the input vector is  x = [x0=-1,x1= 4],, and the bias is  w2  =-2.
    Use PyTorch to calculate the forward pass of the network, return y_hat = f(w,x).
    """
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)
    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    y_hat = torch.tanh(w0 * x0 + w1 * x1 + w2)

    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients 
    for each of the weights, and return the gradient of them 
    in order of w0, w1, and w2.
    """
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)
    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    y_hat = torch.tanh(w0 * x0 + w1 * x1 + w2)

    y = torch.tensor(1.0)
    
    loss = (y_hat - y) ** 2
    loss.backward()

    gw0 = w0.grad
    gw1 = w1.grad
    gw2 = w2.grad

    return gw0, gw1, gw2

def backprop_c():
    """
    Assuming a learning rate of 0.1, 
    update each of the weights accordingly. 
    For simplicity, just do one iteration. 
    And return the updated weights in the order of w0, w1, and w2 
    """
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)
    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    y_hat = torch.tanh(w0 * x0 + w1 * x1 + w2)
    
    y = torch.tensor(1.0)
    
    loss = (y_hat - y) ** 2
    loss.backward()

    learning_rate = 0.1

    w0 = w0 - learning_rate * w0.grad
    w1 = w1 - learning_rate * w1.grad
    w2 = w2 - learning_rate * w2.grad

    return  w0, w1, w2 


def constructParaboloid(w=256, h=256):
    img = np.zeros((w, h), np.float32)
    for x in range(w):
        for y in range(h):
            # let's center the paraboloid in the img
            img[y, x] = (x - w / 2) ** 2 + (y - h / 2) ** 2
    return img


def newtonMethod(x0, y0):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0) 
    paraboloid = torch.unsqueeze(paraboloid, 0)    # -> (1,1,H,W) for conv2d

    """
    Insert your code here
    """
    # Central Difference Kernels
    kx = torch.tensor([[[[-0.5, 0.0,  0.5],
                         [-0.5, 0.0,  0.5],
                         [-0.5, 0.0,  0.5]]]], dtype=torch.float32)   # ∂/∂x

    ky = torch.tensor([[[[-0.5, -0.5, -0.5], 
                         [ 0.0,  0.0,  0.0],
                         [ 0.5,  0.5,  0.5]]]], dtype=torch.float32)  # ∂/∂y
    
    fx = F.conv2d(paraboloid, kx, padding=1)
    fy = F.conv2d(paraboloid, ky, padding=1)

    fxx = F.conv2d(fx, kx, padding=1)
    fyy = F.conv2d(fy, ky, padding=1)
    fxy = F.conv2d(fx, ky, padding=1)

    x = x0
    y = y0

    for _ in range(10):
        # Gradient
        grad_x = fx[0, 0, y, x].item()
        grad_y = fy[0, 0, y, x].item()
        grad = torch.tensor([[grad_x], [grad_y]], dtype=torch.float32)

        # Hessian
        f_xx = fxx[0, 0, y, x].item()
        f_yy = fyy[0, 0, y, x].item()
        f_xy = fxy[0, 0, y, x].item()
        hessian = torch.tensor([[f_xx, f_xy],
                                [f_xy, f_yy]], dtype=torch.float32)

        # Update step
        hessian_inv = torch.inverse(hessian)
        delta = -hessian_inv @ grad

        x += int(round(delta[0].item()))
        y += int(round(delta[1].item()))

        # Clamp to image boundaries
        x = max(0, min(paraboloid.shape[3] - 1, x))
        y = max(0, min(paraboloid.shape[2] - 1, y))

    final_x = x
    final_y = y

    return final_x, final_y


def sgd(x0, y0, lr=0.001):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = torch.unsqueeze(paraboloid, 0)

    """
    Insert your code here
    """
    # Central Difference Kernels
    kx = torch.tensor([[[[-0.5, 0.0,  0.5],
                         [-0.5, 0.0,  0.5],
                         [-0.5, 0.0,  0.5]]]], dtype=torch.float32)   # ∂/∂x

    ky = torch.tensor([[[[-0.5, -0.5, -0.5], 
                         [ 0.0,  0.0,  0.0],
                         [ 0.5,  0.5,  0.5]]]], dtype=torch.float32)  # ∂/∂y
    
    fx = F.conv2d(paraboloid, kx, padding=1)
    fy = F.conv2d(paraboloid, ky, padding=1)

    x = float(x0)
    y = float(y0)

    for _ in range(1000):
        x_int = int(round(x))
        y_int = int(round(y))
        # Gradient
        grad_x = fx[0, 0, y_int, x_int].item()
        grad_y = fy[0, 0, y_int, x_int].item()

        # Update step
        x -= lr * grad_x
        y -= lr * grad_y

        # Clamp to image boundaries
        x = max(0, min(paraboloid.shape[3] - 1, x))
        y = max(0, min(paraboloid.shape[2] - 1, y))

    final_x = int(round(x))
    final_y = int(round(y))

    return final_x, final_y


# if __name__ == "__main__":
#     plt.imshow(constructParaboloid(), cmap='plasma')
#     plt.title("Paraboloid")
#     plt.show() return final_x, final_y