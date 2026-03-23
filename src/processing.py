import numpy as np
from . import io

def convert_encoding(image_array, input_encoding, output_encoding):
    """
    Convierte un array de imagen entre diferentes codificaciones.
    """
    if input_encoding.lower() == output_encoding.lower():
        return image_array

    # Obtener propiedades de las codificaciones
    in_bits, _, in_sign_char = io.parse_encoding(input_encoding)
    out_bits, out_endian, out_sign_char = io.parse_encoding(output_encoding)

    # Usamos la endianness nativa de la máquina para los cálculos
    working_array = image_array.astype(image_array.dtype.newbyteorder('='))

    # --- PASO 1: Convertir la entrada a un formato intermedio SIN SIGNO ---
    if in_sign_char == 'i':
        if in_bits == 16:
            working_array = (working_array.astype(np.int32) + 32768).astype(np.uint16)
        elif in_bits == 8:
            working_array = (working_array.astype(np.int16) + 128).astype(np.uint8)

    # --- PASO 2: Gestionar el cambio de profundidad de bits ---
    if in_bits == 16 and out_bits == 8:
        min_val = np.min(working_array)
        max_val = np.max(working_array)
        
        if max_val == min_val:
            working_array = np.zeros_like(working_array, dtype=np.uint8)
        else:
            float_array = working_array.astype(np.float32)
            normalized = 255 * (float_array - min_val) / (max_val - min_val)
            working_array = normalized.astype(np.uint8)

    elif in_bits == 8 and out_bits == 16:
        working_array = working_array.astype(np.uint16) * 257
    
    # --- PASO 3: Convertir al formato de signo de SALIDA ---
    if out_sign_char == 'i':
        if out_bits == 16:
            working_array = (working_array.astype(np.int32) - 32768)
        elif out_bits == 8:
            working_array = (working_array.astype(np.int16) - 128)

    final_dtype_str = f'{out_endian}{out_sign_char}{out_bits // 8}'
    final_array = working_array.astype(np.dtype(final_dtype_str))
    
    return final_array