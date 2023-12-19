import numpy as np
import matplotlib.pyplot as plt
import noise
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
from PIL import Image
#from arrows3dplot import * # python_file in project with class
import matplotlib.cm as cm
import math 
import random

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(arr)

    normalized_array = -1 + 2 * (arr - min_val) / (max_val - min_val)
    return normalized_array



# Función para generar un terreno inicial aleatorio usando ruido Perlin
def generate_perlin_terrain(size, scale, octaves, persistence, lacunarity, seed):
    #genera un Array 2D de elementos de rango [-1,1] usando perlin noise:
    #Size : las dimensiones del terreno generado, en este caso solo puede ser un cuadrado
    #Scale : el grado de zoom que tendrá el terreno
    #Octave : agrega detalles a las superficie, por ejemplo octave 1 pueden ser las montañas,
    #octave 2 pueden ser las rocas, son como multiples pasadas al terreno para agregarle detalle
    #Lacuranity : ajusta la frequencia en la que se agrega detalle en octave,
    #un valor deseable suele ser 2
    #Persistence : determina la influencia que tiene cada octave
    terrain = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            terrain[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=size, repeaty=size, base=seed)
    return terrain

def generate_random_matrix(size):
    return np.random.uniform(low=-1, high=1, size=(size,size))

# Función de evaluación basada en la pendiente del terreno
def evaluate_terrain(terrain):
    # Calcular la pendiente del terreno

    gradient_x, gradient_y = np.gradient(terrain)
    val = gradient_x**2 + gradient_y**2
     
    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    # En este ejemplo, la función de evaluación es la suma de las pendientes
    return np.sum(slope)

# Simulated Annealing para generar terreno realista
def simulated_annealing(initial_state, max_iterations, cooling_rate):
    current_state = initial_state
    current_energy = evaluate_terrain(current_state)
    iteration_arr = []
    energy_arr = []
    a,b = np.shape(initial_state)

    for iteration in range(max_iterations):
        temperature = initial_temperature / (1 + cooling_rate * iteration)
        # Perturb the current state
        new_state = current_state + np.random.pareto(a=1.8, size = (a,b))
        new_energy = evaluate_terrain(new_state)

        # Calculate the change in energy
        delta_energy = new_energy - current_energy

        # Accept the new state with a probability based on the temperature and energy change
        if delta_energy < 0 or random.uniform(0, 1) < math.exp(-delta_energy / temperature):
            current_state = new_state
            current_energy = new_energy

        # Print or log the energy at regular intervals
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Energy: {current_energy}")

        iteration_arr.append(iteration)
        energy_arr.append(current_energy)
        # Add other termination conditions if needed

    return current_state, iteration_arr , energy_arr

if __name__ == "__main__":
    # Parámetros
    terrain_size = 250
    initial_temperature = 1.0
    max_iterations = 5000
    cooling_rate = 0.01

    # Genera estado inicial y aplica Simulated Annealing
    initial_state = np.random.pareto(a=1.5, size = (terrain_size,terrain_size))
    final_state , iteration_arr, energy_arr = simulated_annealing(initial_state, max_iterations, cooling_rate)

    plt.plot(iteration_arr, energy_arr)
    plt.xlabel('Iteración')
    plt.ylabel('Energía')
    plt.title('Simulated Annealing - resultados de energía')
    plt.yscale('log')
    plt.show()

    # Visualización del terreno inicial
    plt.subplot(1, 2, 1)
    plt.title('Pareto Texture')
    plt.imshow(initial_state, cmap='terrain', interpolation='bilinear')

    # Visualización del terreno final
    plt.subplot(1, 2, 2)
    plt.title('Simulated Annealing \ Pareto Texture')
    plt.imshow(final_state, cmap='terrain', interpolation='bilinear')

    plt.show()
