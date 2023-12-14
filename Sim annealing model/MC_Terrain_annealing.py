import numpy as np
import matplotlib.pyplot as plt
import noise

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(arr)

    normalized_array = -1 + 2 * (arr - min_val) / (max_val - min_val)
    return normalized_array

import numpy as np
import matplotlib.pyplot as plt

def initialize_terrain(size):
    return np.random.rand(size, size)

def markov_chain_terrain(size, iterations, transition_std):
    terrain = initialize_terrain(size)

    for _ in range(iterations):
        new_terrain = np.copy(terrain)

        for x in range(1, size-1):
            for y in range(1, size-1):
                neighborhood = terrain[x-1:x+2, y-1:y+2]
                new_terrain[x, y] = np.mean(neighborhood) + np.random.normal(0, transition_std)

        terrain = new_terrain

    return terrain


# Función de evaluación basada en la pendiente del terreno
def evaluate_terrain(terrain):
    # Calcular la pendiente del terreno

    gradient_x, gradient_y = np.gradient(terrain)
    val = gradient_x**2 + gradient_y**2
     
    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    # En este ejemplo, la función de evaluación es la suma de las pendientes
    return np.sum(slope)

def terrain_cost_proximity(terrain, threshold=0.3, penalty_factor=10, proximity_factor=10):
    # Compute the gradient of the terrain
    gradient_x, gradient_y = np.gradient(terrain)

    # Calculate the magnitude of the gradient vector at each point
    slope = np.sqrt(gradient_x**2 + gradient_y**2)

    # Apply a penalty for slopes above the threshold
    threshold_reward = np.maximum(0, slope - threshold)

    # Calculate penalties based on proximity of values
    proximity_penalty = proximity_factor * np.sum(np.abs(np.diff(terrain, axis=0)))
    proximity_penalty += proximity_factor * np.sum(np.abs(np.diff(terrain, axis=1)))

    # Calculate the overall cost as the sum of penalties
    cost = np.sum(threshold_reward) * penalty_factor - proximity_penalty

    return cost

def cuadrante_normal(size):
    terrain = np.zeros((size, size))

    # Dividir la matriz en cuartos y aplicar distribución normal a cada cuarto
    half_size = size // 2
    terrain[:half_size, :half_size] = np.random.normal(loc=0, scale=1, size=(half_size, half_size))
    terrain[half_size:, half_size:] = np.random.normal(loc=0, scale=1, size=(half_size, half_size))

    return terrain

# Simulated Annealing para generar terreno realista
def simulated_annealing(initial_terrain, iterations, initial_temperature, cooling_rate, terrain_size):
    current_terrain = initial_terrain.copy()
    current_energy = terrain_cost_proximity(current_terrain)

    for iteration in range(iterations):
        # Generar un nuevo terreno vecino
        new_terrain = current_terrain + np.mean(current_terrain)
        new_energy = terrain_cost_proximity(new_terrain)

        # Calcular la diferencia de energía
        energy_difference = new_energy - current_energy

        # Decidir si aceptar el nuevo terreno
        if energy_difference < 0 or np.random.rand() < np.exp(-energy_difference / (initial_temperature - 1e-15)):
            current_terrain = new_terrain
            current_energy = new_energy

        # Enfriar la temperatura
        initial_temperature *= cooling_rate

    return current_terrain

# Parámetros
terrain_size = 100
scale = 20.0
octaves = 6
persistence = 0.5
lacunarity = 2.0
seed = 42
iterations = 1000
initial_temperature = 1.0
cooling_rate = 0.8

# Generar terreno inicial usando ruido Perlin
initial_terrain = markov_chain_terrain(terrain_size, 100, 0.1)

# Aplicar Simulated Annealing
final_terrain = simulated_annealing(initial_terrain, iterations, initial_temperature, cooling_rate, terrain_size)
plt.figure(figsize=(12, 6))

plt.imshow(initial_terrain, cmap='terrain', origin='lower')
plt.colorbar()
plt.title('Terreno Generado con Ruido Perlin')

# Visualizar terreno generado
plt.figure()
plt.imshow(final_terrain, cmap='terrain', origin='lower')
plt.colorbar()
plt.title('Terreno Generado con Simulated Annealing y Ruido Perlin')

plt.tight_layout()
plt.show()