#!/usr/bin/env python3
"""
Interactive 3D Milky Way Galaxy Simulation
==========================================
A realistic real-time simulation of the Milky Way galaxy with:
- Spiral arm structure
- Central bulge
- Stellar halo
- Interactive rotation and zoom controls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

class MilkyWaySimulation:
    def __init__(self, num_stars=50000):
        """
        Initialize the Milky Way simulation

        Parameters:
        -----------
        num_stars : int
            Total number of stars to simulate
        """
        self.num_stars = num_stars
        self.angle = 0
        self.elevation = 30
        self.azimuth = 45

        # Generate galaxy structure
        print("Generating galaxy structure...")
        self.generate_galaxy()

        # Set up the plot
        print("Setting up 3D visualization...")
        self.setup_plot()

    def spiral_arm(self, theta, arm_offset, num_points, spread=0.3):
        """
        Generate a logarithmic spiral arm

        Parameters:
        -----------
        theta : array
            Angular positions
        arm_offset : float
            Angular offset for this spiral arm
        num_points : int
            Number of stars in this arm
        spread : float
            Spread/thickness of the arm
        """
        # Logarithmic spiral: r = a * e^(b*theta)
        a = 0.5
        b = 0.3

        theta = theta + arm_offset
        r = a * np.exp(b * theta)

        # Add randomness for natural look
        r += np.random.normal(0, spread, num_points)
        theta_rand = theta + np.random.normal(0, 0.1, num_points)

        # Convert to Cartesian coordinates
        x = r * np.cos(theta_rand)
        y = r * np.sin(theta_rand)

        # Add vertical spread (disk thickness)
        z = np.random.normal(0, 0.1, num_points)

        return x, y, z, r

    def generate_galaxy(self):
        """Generate the entire galaxy structure"""

        # Distribution of stars across components
        num_disk = int(self.num_stars * 0.7)  # 70% in disk/spiral arms
        num_bulge = int(self.num_stars * 0.2)  # 20% in central bulge
        num_halo = int(self.num_stars * 0.1)   # 10% in halo

        # Generate spiral arms (4 major arms)
        num_per_arm = num_disk // 4
        theta = np.linspace(0, 4 * np.pi, num_per_arm)

        x_disk, y_disk, z_disk, r_disk = [], [], [], []

        for arm_offset in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x, y, z, r = self.spiral_arm(theta, arm_offset, num_per_arm)
            x_disk.extend(x)
            y_disk.extend(y)
            z_disk.extend(z)
            r_disk.extend(r)

        self.x_disk = np.array(x_disk)
        self.y_disk = np.array(y_disk)
        self.z_disk = np.array(z_disk)
        self.r_disk = np.array(r_disk)

        # Generate central bulge (spherical distribution)
        r_bulge = np.random.exponential(1.5, num_bulge)
        theta_bulge = np.random.uniform(0, 2*np.pi, num_bulge)
        phi_bulge = np.random.uniform(0, np.pi, num_bulge)

        self.x_bulge = r_bulge * np.sin(phi_bulge) * np.cos(theta_bulge)
        self.y_bulge = r_bulge * np.sin(phi_bulge) * np.sin(theta_bulge)
        self.z_bulge = r_bulge * np.cos(phi_bulge)

        # Generate halo (spherical, larger radius)
        r_halo = np.random.exponential(15, num_halo)
        theta_halo = np.random.uniform(0, 2*np.pi, num_halo)
        phi_halo = np.random.uniform(0, np.pi, num_halo)

        self.x_halo = r_halo * np.sin(phi_halo) * np.cos(theta_halo)
        self.y_halo = r_halo * np.sin(phi_halo) * np.sin(theta_halo)
        self.z_halo = r_halo * np.cos(phi_halo)

    def get_star_colors_and_sizes(self):
        """Assign colors and sizes based on stellar population"""

        # Disk stars - blue-white (young hot stars)
        colors_disk = np.array(['#87CEEB'] * len(self.x_disk))
        sizes_disk = np.random.uniform(0.1, 1.5, len(self.x_disk))

        # Bulge stars - yellow-orange (older stars)
        colors_bulge = np.array(['#FFD700'] * len(self.x_bulge))
        sizes_bulge = np.random.uniform(1.0, 3.0, len(self.x_bulge))

        # Halo stars - red (old population II stars)
        colors_halo = np.array(['#FF6B6B'] * len(self.x_halo))
        sizes_halo = np.random.uniform(0.5, 1.0, len(self.x_halo))

        return (colors_disk, colors_bulge, colors_halo,
                sizes_disk, sizes_bulge, sizes_halo)

    def setup_plot(self):
        """Set up the matplotlib 3D plot"""

        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')

        # Get colors and sizes
        (colors_disk, colors_bulge, colors_halo,
         sizes_disk, sizes_bulge, sizes_halo) = self.get_star_colors_and_sizes()

        # Plot each component
        self.scatter_disk = self.ax.scatter(
            self.x_disk, self.y_disk, self.z_disk,
            c=colors_disk, s=sizes_disk, alpha=0.6, label='Disk Stars'
        )

        self.scatter_bulge = self.ax.scatter(
            self.x_bulge, self.y_bulge, self.z_bulge,
            c=colors_bulge, s=sizes_bulge, alpha=0.8, label='Bulge Stars'
        )

        self.scatter_halo = self.ax.scatter(
            self.x_halo, self.y_halo, self.z_halo,
            c=colors_halo, s=sizes_halo, alpha=0.3, label='Halo Stars'
        )

        # Set labels and title
        self.ax.set_xlabel('X (kpc)', color='white', fontsize=10)
        self.ax.set_ylabel('Y (kpc)', color='white', fontsize=10)
        self.ax.set_zlabel('Z (kpc)', color='white', fontsize=10)
        self.ax.set_title('Interactive 3D Milky Way Galaxy Simulation\n' +
                         'Use mouse to rotate | Scroll to zoom',
                         color='white', fontsize=14, pad=20)

        # Set axis properties
        max_range = 30
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])

        # Style the axes
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        self.ax.grid(True, alpha=0.2, color='gray')

        # Add legend
        legend = self.ax.legend(loc='upper left', fontsize=10, framealpha=0.3)
        for text in legend.get_texts():
            text.set_color("white")

        # Add info text
        info_text = (
            f"Total Stars: {self.num_stars:,}\n"
            f"Disk: {len(self.x_disk):,} | "
            f"Bulge: {len(self.x_bulge):,} | "
            f"Halo: {len(self.x_halo):,}"
        )
        self.fig.text(0.5, 0.02, info_text, ha='center',
                     color='white', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def animate(self, frame):
        """Animation function for rotation"""
        self.angle += 1
        self.ax.view_init(elev=self.elevation, azim=self.angle)
        return self.scatter_disk, self.scatter_bulge, self.scatter_halo

    def start_animation(self, interval=50):
        """Start the animated rotation"""
        print("Starting animation...")
        self.anim = FuncAnimation(
            self.fig, self.animate, interval=interval,
            blit=False, cache_frame_data=False
        )
        plt.show()

    def show_static(self):
        """Show static interactive plot (no auto-rotation)"""
        print("Displaying interactive 3D plot...")
        print("- Use mouse to rotate the view")
        print("- Scroll to zoom in/out")
        print("- Click and drag to change perspective")
        plt.show()


def main():
    """Main entry point"""
    print("=" * 60)
    print("   MILKY WAY GALAXY - 3D INTERACTIVE SIMULATION")
    print("=" * 60)
    print()

    # Ask user for preferences
    print("Choose simulation mode:")
    print("1. Static interactive (recommended - full mouse control)")
    print("2. Auto-rotating animation")
    print()

    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
    if not choice:
        choice = "1"

    # Number of stars
    num_stars_input = input("Enter number of stars (10000-100000) [default: 50000]: ").strip()
    if num_stars_input and num_stars_input.isdigit():
        num_stars = int(num_stars_input)
        num_stars = max(10000, min(100000, num_stars))
    else:
        num_stars = 50000

    print()
    print(f"Initializing simulation with {num_stars:,} stars...")
    print()

    # Create and run simulation
    galaxy = MilkyWaySimulation(num_stars=num_stars)

    if choice == "2":
        galaxy.start_animation()
    else:
        galaxy.show_static()


if __name__ == "__main__":
    main()
