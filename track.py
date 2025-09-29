# track.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_track_from_csv(path):
    """
    Load a racetrack grid from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        np.ndarray: 2D array representing the track.
    """
    return np.loadtxt(path, delimiter=",", dtype=int)

class Track:
    """
    Management of racetrack properties and visualization.
    """
    def __init__(self, grid):
        """
           grid: 2D ndarray which rapresents the map
             0 = invalid cell (outside the track)
             1 = valid cell (inside the track)
             2 = start
             3 = finish
        """
        self.grid = grid
        self.heigth, self.width = grid.shape

       
        self.start_cells = [(x, y) for x in range(self.heigth)  
                                     for y in range(self.width) 
                                     if grid[x, y] == 2]
        self.finish_cells = [(x, y) for x in range(self.heigth) 
                                      for y in range(self.width)
                                      if grid[x, y] == 3]

    def is_inside(self, x, y):
        """
           True if the racecar is inside the track contours
        """
        return (0 <= x < self.heigth and 0 <= y < self.width
                and self.grid[x, y] != 0)

    def is_start(self, x, y):
        """
           True if the racecar is on the starting line
        """
        return (x, y) in self.start_cells

    def is_finish(self, x, y):
        """
           True if the racecar is on the finish line
        """
        return (x, y) in self.finish_cells

    def random_start(self):
        """
           Returns a random (x,y) from starting cells 
        """
        return self.start_cells[np.random.randint(len(self.start_cells))]


    def display(self):
        """
           Displays the track as a color gridmap
        """
        
        cmap = ListedColormap([
            "black",   # 0
            "white",   # 1
            "red",     # 2
            "green"    # 3
        ])

        plt.imshow(self.grid, cmap=cmap, origin="upper")

        plt.xticks([])
        plt.yticks([])

        plt.grid(True, color="gray", linewidth=0.5)
        plt.gca().set_xticks(np.arange(-.5, self.grid.shape[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, self.grid.shape[0], 1), minor=True)
        plt.grid(which="minor", color="gray", linewidth=0.5)
        plt.gca().tick_params(which="minor", bottom=False, left=False)

        plt.show()

