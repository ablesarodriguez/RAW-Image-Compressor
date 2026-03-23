import numpy as np
import math

def calculate_metrics(original_array, processed_array, bits):
    """
    Calcula las métricas PAE, MSE y PSNR entre dos imágenes.

    Args:
        original_array (np.array): El array de la imagen original.
        processed_array (np.array): El array de la imagen procesada (ej. dequantizada).
        bits (int): La profundidad de bits de la imagen ORIGINAL (ej. 8, 16).

    Returns:
        dict: Un diccionario con las métricas 'PAE', 'MSE' y 'PSNR'.
    """
    original_float = original_array.astype(np.float64)
    processed_float = processed_array.astype(np.float64)
    
    # --- PAE (Peak Absolute Error) ---
    abs_error = np.abs(original_float - processed_float)
    pae = np.max(abs_error)
    
    # --- MSE (Mean Squared Error) ---
    squared_error = (original_float - processed_float) ** 2
    mse = np.mean(squared_error)
    
    # --- PSNR (Peak Signal-to-Noise Ratio) ---
    if mse == 0:
        psnr = np.inf
    else:
        max_i = (2**bits) - 1
        
        # Fórmula PSNR: 10 * log10( (MAX_I^2) / MSE )
        psnr = 10 * math.log10((max_i**2) / mse)
        
    return {
        'PAE': pae,
        'MSE': mse,
        'PSNR': psnr
    }