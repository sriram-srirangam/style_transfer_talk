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