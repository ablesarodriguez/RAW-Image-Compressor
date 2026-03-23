import numpy as np
from pathlib import Path

def parse_encoding(encoding_str):
    """
    Analiza la cadena para determinar bits, endianness y si es con signo.
    Devuelve: (bits, endianness_char, signed_char)
    """
    encoding_lower = encoding_str.lower()
    bits = 0
    endianness_char = '<'  # Little-endian por defecto
    signed_char = 'u'      # Unsigned por defecto

    if 'ube' in encoding_lower:
        endianness_char = '>'
    elif 'sbe' in encoding_lower:
        endianness_char = '>'
        signed_char = 'i'  # 'i' para integer (signed)
    elif 'ule' in encoding_lower:
        endianness_char = '<'
    elif 'sle' in encoding_lower:
        endianness_char = '<'
        signed_char = 'i'

    if '16' in encoding_lower or '2' in encoding_lower:
        bits = 16
    elif '8' in encoding_lower or '1' in encoding_lower:
        bits = 8

    if bits == 0:
        raise ValueError(f"No se pudo determinar el número de bits desde la codificación: '{encoding_str}'")

    return bits, endianness_char, signed_char


def load_raw_image(filepath, width, height, channels, encoding):
    """
    Carga una única imagen RAW desde un archivo.
    Versión corregida para manejar tipos con signo.
    """
    try:
        bits, endianness_char, signed_char = parse_encoding(encoding)
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            raw_data = f.read()

        dtype_str = f'{endianness_char}{signed_char}{bits // 8}'
        image_array = np.frombuffer(raw_data, dtype=np.dtype(dtype_str))
        
        expected_pixels = width * height * channels
        if image_array.size != expected_pixels:
            print(f"Advertencia: El tamaño del archivo ({image_array.size} px) no coincide con el esperado ({expected_pixels} px). Se truncarán datos.")
            image_array = image_array[:expected_pixels]

        shape = (height, width, channels) if channels > 1 else (height, width)
        image_array = image_array.reshape(shape)

        return {
            'array': image_array,
            'params': {
                'width': width, 'height': height, 'channels': channels,
                'bits': bits, 'endianness': 'big' if endianness_char == '>' else 'little',
                'encoding': encoding
            }
        }
    except FileNotFoundError:
        print(f"Error: El archivo '{filepath}' no existe.")
        return None
    except Exception as e:
        print(f"Error procesando '{filepath}': {e}")
        return None


def save_raw_image(image_array, filepath):
    """
    Guarda un array de numpy como un archivo de imagen RAW.
    """
    try:
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_array.tofile(output_path)
        print(f"Imagen guardada correctamente en '{output_path}'")
        return True
    except Exception as e:
        print(f"Error al guardar la imagen en '{filepath}': {e}")
        return False