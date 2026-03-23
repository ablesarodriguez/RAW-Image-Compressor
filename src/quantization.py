import numpy as np

def quantize_image(image_array, q_step):
    """
    Args:
        image_array (np.array): El array de la imagen original.
        q_step (float): El tamaño del paso de cuantización (Δ).

    Returns:
        np.array: Un array con los índices cuantizados.
    """
    if q_step <= 0:
        raise ValueError("q_step debe ser un número positivo.")

    float_array = image_array.astype(np.float64)
    
    # MEJORA 1: Usar redondeo aritmético (floor(x + 0.5)) en lugar de Bankers Rounding.
    # Esto es más consistente para imágenes.
    # Preservamos el signo para que funcione correctamente con residuos negativos si fuera necesario.
    abs_val = np.abs(float_array)
    quantized_indices = np.floor(abs_val / q_step + 0.5) * np.sign(float_array)
   
    return quantized_indices.astype(np.int16)


def dequantize_image(quantized_array, q_step, bits=8):
    """
    Args:
        quantized_array (np.array): Índices cuantizados.
        q_step (float): Paso de cuantización.
        bits (int): Profundidad de bits original (para hacer clipping).

    Returns:
        np.array: Valores reconstruidos.
    """
    if q_step <= 0:
        raise ValueError("q_step debe ser un número positivo.")

    float_array = quantized_array.astype(np.float64)
    dequantized_float = float_array * q_step
    
    # MEJORA 2: Redondear antes de convertir a entero.
    # Evita el truncamiento (ej: 9.9 se convertía en 9, ahora en 10).
    dequantized_values = np.round(dequantized_float)
    
    # MEJORA 3 (LA MÁS IMPORTANTE): Clipping.
    # Si la reconstrucción se sale del rango real (ej: 260 en 8-bit), 
    # la recortamos a 255. Esto reduce drásticamente el error (MSE).
    max_val = (2**bits) - 1
    np.clip(dequantized_values, 0, max_val, out=dequantized_values)
    
    return dequantized_values.astype(np.int16)