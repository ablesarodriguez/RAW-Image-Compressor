import json
from . import io
from . import quantization
from . import predictor
from . import entropy
from . import arithmetic

def encode_image(inputpath, outputpath, width, height, channels, encoding, qstep, mode):
    print(f"--- Iniciando codificación: {inputpath} ---")

    # 1. Cargar
    image_data = io.load_raw_image(inputpath, width, height, channels, encoding)
    if not image_data: return False
    
    original_array = image_data['array']
    original_params = image_data['params']

    # 2. Cuantizar
    quantized_indices = quantization.quantize_image(original_array, qstep)

    # 3. Predecir
    final_residue = predictor.predict_image(quantized_indices, mode)

    # 4. Frecuencias
    symbol_counts = entropy.calculate_frequencies(final_residue.flatten())
    symbol_counts_serializable = {str(k): v for k, v in symbol_counts.items()}
    total_symbols = final_residue.size

    # 5. Header
    header_data = {
        'width': width, 'height': height, 'channels': channels,
        'original_encoding': encoding, 'original_bits': original_params['bits'],
        'qstep': qstep, 'predictor_mode': mode,
        'total_symbols': total_symbols, 'frequencies': symbol_counts_serializable
    }
    header_json = json.dumps(header_data, indent=4)
    header_bytes = header_json.encode('utf-8')

    # 6. Aritmética
    bitstream_bytes = arithmetic.encode(final_residue, symbol_counts_serializable, total_symbols)

    # 7. Guardar
    try:
        with open(outputpath, 'wb') as f:
            f.write(len(header_bytes).to_bytes(8, byteorder='little'))      
            f.write(header_bytes)
            f.write(bitstream_bytes)
        print(f"¡Codificación exitosa! Guardado en: {outputpath}")
        return True
    except Exception as e:
        print(f"Error guardando archivo: {e}")
        return False

def decode_image(inputpath, outputpath):
    print(f"--- Iniciando decodificación: {inputpath} ---")
    try:
        with open(inputpath, 'rb') as f:
            # 1. Header
            header_len_bytes = f.read(8)
            header_length = int.from_bytes(header_len_bytes, byteorder='little')
            header_json = f.read(header_length).decode('utf-8')
            header_data = json.loads(header_json)
            
            # 2. Bitstream
            bitstream_bytes = f.read()

            # Parámetros
            width, height, channels = header_data['width'], header_data['height'], header_data['channels']
            qstep, mode = header_data['qstep'], header_data['predictor_mode']
            total_symbols = header_data['total_symbols']
            freqs = header_data['frequencies']
            original_bits = header_data.get('original_bits', 8) # Fallback a 8 si no existe

            # 3. Decodificar Aritmética
            flat_residue = arithmetic.decode(bitstream_bytes, freqs, total_symbols)
            
            shape = (height, width, channels) if channels > 1 else (height, width)
            mapped_residue_array = flat_residue.reshape(shape)

            # 4. Reconstruir (Inversa Predictor)
            reconstructed_indices = predictor.reconstruct_image(mapped_residue_array, mode, original_bits)

            # 5. Decuantizar
            dequantized_array = quantization.dequantize_image(reconstructed_indices, qstep, bits=original_bits)

            # 6. Guardar
            io.save_raw_image(dequantized_array, outputpath)
            print(f"¡Decodificación exitosa! Guardado en: {outputpath}")
            return True

    except Exception as e:
        print(f"Error durante la decodificación: {e}")
        return False