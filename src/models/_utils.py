import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

def render_vid(imgs):
    fig = plt.figure()
    ax = plt.axes()

    im = ax.imshow(imgs[0])

    def animate(i):
        im.set_data(imgs[i])
        return im,

    anim = FuncAnimation(fig, animate, frames=len(imgs), interval=100)
    return anim