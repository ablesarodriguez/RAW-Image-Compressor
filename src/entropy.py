from collections import Counter
import math

def calculate_frequencies(array):
    return Counter(array)

def calculate_entropy(image_array):
    """
    Calcula la entropía de orden 0 y orden 1 de una imagen.

    Args:
        image_array (np.array): El array de la imagen.

    Returns:
        (float, float): Tupla con (entropía_orden_0, entropía_orden_1).
    """
    if image_array is None or image_array.size == 0:
        return 0.0, 0.0

    # --- Entropía de orden 0 ---
    # Se calcula sobre los valores de píxeles individuales
    flat_array = image_array.flatten()
    total_pixels = flat_array.size
    value_counts = calculate_frequencies(flat_array)
    
    entropy_0 = 0.0
    for count in value_counts.values():
        probability = count / total_pixels
        entropy_0 -= probability * math.log2(probability)

    # --- Entropía de orden 1 ---
    # Entropía condicional: H(X_i | X_{i-1})
    # Se basa en la probabilidad de un píxel dado su vecino anterior
    pairs = list(zip(flat_array, flat_array[1:]))
    pair_counts = calculate_frequencies(pairs)
    prev_pixel_counts = calculate_frequencies(flat_array[:-1])

    entropy_1 = 0.0
    for pair, count in pair_counts.items():
        prev_pixel = pair[0]
        joint_prob = count / (total_pixels - 1)
        conditional_prob = count / prev_pixel_counts[prev_pixel]
        entropy_1 -= joint_prob * math.log2(conditional_prob)

    return entropy_0, entropy_1