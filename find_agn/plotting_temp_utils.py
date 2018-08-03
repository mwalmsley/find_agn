import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
https://stackoverflow.com/questions/6309472/matplotlib-can-i-create-axessubplot-objects-then-add-them-to-a-figure-instance
You would first need to remove the axes from the first figure, 
then append it to the next figure and give it some position to live in. 
"""

def copy_to_dummy_ax(ax, dummy_ax, fig):
    copy_axes(ax, fig)
    ax.set_position(dummy_ax.get_position())
    dummy_ax.remove()


def copy_axes(ax, fig):
    ax.remove()
    ax.figure = fig
    fig.axes.append(ax)
    fig.add_axes(ax)

if __name__ == '__main__':
    fig1, ax = plt.subplots()
    ax.plot(range(10))
    fig2 = plt.figure()
    fig2, (dummy_ax0, dummy_ax1) = plt.subplots(nrows=2)

    copy_to_dummy_ax(ax, dummy_ax1, fig2)

    fig2.savefig('scratch.png')
