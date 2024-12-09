import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
mpl.use('macosx')


class CurveDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Draw a curve (click and drag)")
        self.line, = self.ax.plot([], [], 'r-')
        self.points = []
        self.drawing = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.button = Button(self.button_ax, 'Done')
        self.button.on_clicked(self.on_done)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.points = [[event.xdata, event.ydata]]
        self.line.set_data(*zip(*self.points))
        self.fig.canvas.draw()

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        self.points.append([event.xdata, event.ydata])
        self.line.set_data(*zip(*self.points))
        self.fig.canvas.draw()

    def on_release(self, event):
        self.drawing = False

    def on_done(self, event):
        plt.close()

    def draw(self):
        plt.show()

    def sample_and_save(self):
        if len(self.points) < 2:
            print("Not enough points to create a curve.")
            return None

        # Convert points to numpy array
        points_array = np.array(self.points)

        # Sample 100 uniformly spaced points based on indices
        if len(points_array) >= 100:
            indices = np.linspace(0, len(points_array) - 1, 100, dtype=int)
            sampled_points = points_array[indices]
        else:
            # If less than 100 points, use all points and pad with the last point
            sampled_points = np.pad(points_array, ((0, 100 - len(points_array)), (0, 0)), mode='edge')

        # Save the array
        np.save('curve_data.npy', sampled_points)
        print("Curve data saved as 'curve_data.npy'")
        print("Shape of sampled_points:", sampled_points.shape)

        return sampled_points

# Usage
drawer = CurveDrawer()
drawer.draw()
curve_data = drawer.sample_and_save()

if curve_data is not None:
    # Plot the sampled curve
    plt.figure()
    plt.plot(curve_data[:, 0], curve_data[:, 1], 'b-')
    plt.title("Sampled Curve (100 points)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()