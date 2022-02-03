import torch

import numpy as np
import matplotlib.pyplot as plt

def with_grad(torch_output):
    torch_output.backward()
    return torch_output

def plot_points(x, y, title, color="black", grid=True, legend=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    plt.plot(x, y, label=title, color=tuple(np.random.choice(range(80), size=3) / 255))
    plt.title(title, fontweight="bold")
    
    if grid:
        plt.grid()
        
    if legend:
        plt.legend()

    return fig

def gradient_descent(f, start, lr=0.01, eps=0.01):
    p = start
    p = p.clone().detach().requires_grad_(True)
    f_p = f(p)
    dp_dx = p.grad

    ps = [p.data]
    f_ps = [f_p]
    grads = [dp_dx]
    
    # Keep iterating until gradient becomes arbitrarily small
    while torch.abs(dp_dx) > eps:

        # Step down the tangent line
        p = p - lr * dp_dx

        # Calculate function value/gradient at new point
        p = p.clone().detach().requires_grad_(True)
        f_p = f(p)
        dp_dx = p.grad
        
        # Bookkeeping shit
        ps.append(p.data)
        f_ps.append(f_p)
        grads.append(dp_dx)
    
    return ps, f_ps, grads
