import numpy as np
import cupy as cp  # This requires CuPy for GPU-based computations

class ElectromagneticWaveSimulation:
    """
    Class to simulate electromagnetic waves using GPU programming (using CuPy for demonstration).
    """

    def __init__(self, grid_size=(100, 100), time_steps=100, wave_type="electromagnetic"):
        """
        Initialize the simulation with specified grid size, time steps, and wave type.
        
        Parameters:
        - grid_size: tuple, specifying the dimensions (2D or 3D) for the simulation grid.
        - time_steps: int, number of time steps to simulate.
        - wave_type: str, type of wave being simulated.
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.wave_type = wave_type
        self.simulation_data = None  # To store simulation results

    def setup_simulation(self):
        """
        Initializes the simulation parameters, sets up the GPU environment.
        """
        dimensions = '3D' if len(self.grid_size) == 3 else '2D'
        print(f"Setting up {self.wave_type} wave simulation on a {dimensions} grid of size {self.grid_size} with {self.time_steps} time steps.")
        # Initialize the wave field as a 2D or 3D grid
        self.simulation_data = cp.zeros(self.grid_size)  # Use CuPy array for GPU computations

    def run_simulation(self):
        """
        Runs the electromagnetic wave simulation on the GPU.
        """
        print("Running GPU simulation...")
        # Example wave propagation simulation (place actual GPU code here)
        for t in range(self.time_steps):
            # Example of updating wave field over time; replace with actual algorithm
            self.simulation_data += cp.random.randn(*self.grid_size) * 0.01
        print("Simulation complete.")

    def save_simulation_results(self, file_name="simulation_results.txt"):
        """
        Saves simulation results to a file.
        """
        simulation_results_cpu = cp.asnumpy(self.simulation_data)  # Move data back to CPU for saving
        np.savetxt(file_name, simulation_results_cpu.reshape(-1))  # Flatten for saving as 1D
        print(f"Simulation results saved to {file_name}")

    def generate_report(self):
        """
        Generates a theoretical and practical report for the assignment.
        """
        report_content = (
            "### State-of-the-Art Approach in Electromagnetic Wave Simulation\n\n"
            "Electromagnetic wave simulations are vital for various applications, "
            "such as radar, communication, and material analysis. This report implements "
            "a GPU-accelerated simulation approach for rapid calculations.\n\n"
            "### Questions & Answers\n\n"
            "1. **Array Processing on CPU vs GPU:**\n"
            "   - **CPU Order:** Row-column order is optimal for CPU due to cache locality.\n"
            "   - **GPU Order:** Column-row order might be preferable due to GPU memory access patterns.\n\n"
            "2. **SIMD vs MIMD Implementations:**\n"
            "   - **SIMD (Single Instruction, Multiple Data):** Efficient for identical operations on large data arrays, as in GPUs.\n"
            "   - **MIMD (Multiple Instruction, Multiple Data):** Allows diverse tasks but requires more complex resource management.\n\n"
            "3. **Memory Choice for Data Structure:**\n"
            "   - For data with a unified access pattern per warp, shared memory is optimal in this scenario to leverage shared data caching.\n"
        )
        with open("report.txt", "w") as file:
            file.write(report_content)
        print("Report generated and saved as 'report.txt'")


# Usage Example
# For 2D simulation
simulation_2d = ElectromagneticWaveSimulation(grid_size=(100, 100), time_steps=100)
simulation_2d.setup_simulation()
simulation_2d.run_simulation()
simulation_2d.save_simulation_results("simulation_results_2d.txt")
simulation_2d.generate_report()

# For 3D simulation
simulation_3d = ElectromagneticWaveSimulation(grid_size=(100, 100, 100), time_steps=100)
simulation_3d.setup_simulation()
simulation_3d.run_simulation()
simulation_3d.save_simulation_results("simulation_results_3d.txt")
simulation_3d.generate_report()
