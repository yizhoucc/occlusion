import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as patches
import matplotlib
import mpl_toolkits.mplot3d.art3d as art3d
BASE_PATH = "/Users/yc/Desktop"

# Arena, Mouse, Aperture classes
class Arena:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height

class Mouse:
    def __init__(self, x, y, z):
        self.position = [x, y, z]

    def move(self, dx, dy, dz):
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz

class Aperture:
    def __init__(self, arena_width, arena_height, arena_length, gap_width):

        self.height = arena_height
        self.width = arena_width/2 - gap_width
        self.gap_width = gap_width
        self.wall_depth = arena_length - 10

        self.left_wall_edge = (self.width, self.wall_depth)
        self.right_wall_edge = (arena_width/2 + gap_width, self.wall_depth)
        #aperature left wall bottom left coordinate
        self.left_wall = (0, self.wall_depth)

        #aperature right wall bottom left coordinate
        self.right_wall = (arena_width/2 + gap_width, self.wall_depth)

# Visualization function with arrow key event handling
def visualize_arena(arena, mouse, aperture):
    #plt.close('all')
    #plt.ion()  # Turn on interactive mode

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot arena boundaries
    ax.set_xlim([0, arena.length])
    ax.set_ylim([0, arena.width])
    ax.set_zlim([0, arena.height])
    
    # Plot mouse position and store the scatter plot object
    mouse_plot = ax.scatter(*mouse.position, color='blue', s=100, label='Mouse')

    # Plot left aperture wall
    left_wall = patches.Rectangle(aperture.left_wall, aperture.width, -aperture.height, color='black', alpha=0.7)
    ax.add_patch(left_wall)
    art3d.pathpatch_2d_to_3d(left_wall, z=arena.width-10, zdir="y")
    
    # Plot right aperture wall
    right_wall = patches.Rectangle(aperture.right_wall, aperture.width, -aperture.height, color='black', alpha=0.7)
    ax.add_patch(right_wall)
    art3d.pathpatch_2d_to_3d(right_wall, z=arena.width-10, zdir="y")

    # Plot first circle on the left side
    circle1 = patches.Circle(( - 10, arena.height/2), 5, color='green')
    ax.add_patch(circle1)
    art3d.pathpatch_2d_to_3d(circle1, z=arena.length, zdir="y")

    # Plot second circle on the right side
    circle2 = patches.Circle((+ 10, arena.height/2), 5, color='red')
    ax.add_patch(circle2)
    art3d.pathpatch_2d_to_3d(circle2, z=arena.length, zdir="y")
    
    info_map = np.load(BASE_PATH + 'info_matrix.npy')
    rows, cols = info_map.shape

    # Create a grid of x and y coordinates
    x = np.linspace(0, cols, cols)
    y = np.linspace(0, rows, rows)
    x, y = np.meshgrid(x, y)

    info_d = info_map # change to desired fourth dimension
    minn, maxx = info_d.min(), info_d.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    m.set_array([])
    fcolors = m.to_rgba(info_d)

# Plot the heatmap in the XY plane (Z = 0)
    ax.plot_surface(x, y, np.zeros_like(info_map), facecolors=fcolors)

    
    ax.set_xlabel('X Axis (Width)')
    ax.set_ylabel('Y Axis (Length)')
    ax.set_zlabel('Z Axis (Height)')
    plt.legend()

    # Function to update the mouse position
    # def update_mouse_position():
    #     nonlocal mouse_plot  # Ensure we use the outer scope variable
    #     # Remove the previous mouse plot
    #     mouse_plot.remove()
    #     # Plot updated mouse position
    #     mouse_plot = ax.scatter(*mouse.position, color='blue', s=100, label='Mouse')
    #     plt.draw()

    # # Event handler for key presses
    # def on_key(event):
    #     print(f"Key pressed: {event.key}")  # Debugging print to confirm keypress is captured
    #     # Step size for movement
    #     step_size = 1
    #     if event.key == 'up':         # Move mouse along the positive Y-axis
    #         mouse.move(0, step_size, 0)
    #     elif event.key == 'down':     # Move mouse along the negative Y-axis
    #         mouse.move(0, -step_size, 0)
    #     elif event.key == 'left':     # Move mouse along the negative X-axis
    #         mouse.move(-step_size, 0, 0)
    #     elif event.key == 'right':    # Move mouse along the positive X-axis
    #         mouse.move(step_size, 0, 0)
    #     elif event.key == 'w':        # Move mouse up along Z-axis
    #         mouse.move(0, 0, step_size)
    #     elif event.key == 's':        # Move mouse down along Z-axis
    #         mouse.move(0, 0, -step_size)

    #     # Update the mouse's position on the plot
    #     update_mouse_position()

    # # Connect the event handler to the figure
    # fig.canvas.mpl_connect('key_press_event', on_key)

    # Show the plot
    plt.show()

