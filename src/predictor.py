import numpy as np

def predict_image(image_array, mode):
    """
    Genera una imagen de residuo restando una predicción de la imagen original (o cuantizada).

    Args:
        image_array (np.array): El array de la imagen original (o cuantizada).
        mode (int): El modo de predicción (1-5).

    Returns:
        np.array: Un array con los residuos mapeados a uint16.
    """
    print(f"-> Aplicando predictor modo {mode}...")
    original = image_array.astype(np.int32)
    
    # --- MODOS VECTORIZADOS (RÁPIDOS) ---
    if mode in [1, 2, 3, 5]:
        prediction = np.zeros_like(original, dtype=np.int32)

        if mode == 1:
            # P = X_{i-1} (vecino horizontal)
            prediction[:, 1:] = original[:, :-1]
        
        elif mode == 2:
            # P = X_{i-W} (vecino vertical)
            prediction[1:, :] = original[:-1, :]
            
        elif mode == 3:
            # P = (X_{i-1} + X_{i-W}) / 2
            pred_h = np.zeros_like(original, dtype=np.int32)
            pred_v = np.zeros_like(original, dtype=np.int32)
            pred_h[:, 1:] = original[:, :-1]
            pred_v[1:, :] = original[:-1, :]
            prediction = (pred_h + pred_v) // 2
            
        elif mode == 5:
            # --- MODO 5: PAETH (Estándar PNG) - Vectorizado ---
            # Es excelente para reducir BPS porque se adapta a los bordes.
            
            # 1. Preparamos matrices desplazadas (Vecinos)
            # a = Izquierda, b = Arriba, c = Diagonal (Arriba-Izquierda)
            a = np.zeros_like(original, dtype=np.int32)
            b = np.zeros_like(original, dtype=np.int32)
            c = np.zeros_like(original, dtype=np.int32)
            
            a[:, 1:] = original[:, :-1]      # Shift Derecha
            b[1:, :] = original[:-1, :]      # Shift Abajo
            c[1:, 1:] = original[:-1, :-1]   # Shift Diagonal

            # 2. Algoritmo Paeth Vectorizado
            # Estimación inicial: p = a + b - c
            p = a + b - c
            
            # Distancias a la estimación
            pa = np.abs(p - a)
            pb = np.abs(p - b)
            pc = np.abs(p - c)
            
            # 3. Selección vectorizada usando máscaras (sin bucles)
            # Si pa <= pb Y pa <= pc -> Elegimos a
            mask_a = (pa <= pb) & (pa <= pc)
            
            # Si no es a, Y pb <= pc -> Elegimos b
            mask_b = (~mask_a) & (pb <= pc)
            
            # Si no es a ni b -> Elegimos c
            # (No necesitamos mask_c explícita, es el resto)
            
            prediction[mask_a] = a[mask_a]
            prediction[mask_b] = b[mask_b]
            # Los que no son A ni B, son C:
            mask_c = ~(mask_a | mask_b)
            prediction[mask_c] = c[mask_c]

        residue = original - prediction
        
    elif mode == 4:
        # Modo Adaptativo (Mínimo Error de Vecino) - Bucle secuencial.
        print("   -> Modo 4: Predictor Adaptativo (Mínimo Error de Vecino).")
        residue = np.zeros_like(original, dtype=np.int32)

        if len(original.shape) == 2:
            height, width = original.shape
            channels = 1
            original_reshaped = original.reshape(height, width, 1)
            residue = residue.reshape(height, width, 1)
        else:
            height, width, channels = original.shape
            original_reshaped = original
            
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    
                    # Vecinos del píxel actual X_curr
                    p_h = original_reshaped[y, x-1, c] if x > 0 else 0
                    p_v = original_reshaped[y-1, x, c] if y > 0 else 0
                    p_diag = original_reshaped[y-1, x-1, c] if x > 0 and y > 0 else 0

                    X_curr = original_reshaped[y, x, c]
                    P_best = 0
                    
                    # --- Lógica del Modo Adaptativo ---
                    P_diag_1 = original_reshaped[y-1, x-2, c] if x > 1 and y > 0 else 0 
                    P_diag_2 = original_reshaped[y-2, x-1, c] if y > 1 and x > 0 else 0 
                    P_diag_3 = (P_diag_1 + P_diag_2) // 2

                    errors = [
                        np.abs(p_diag - P_diag_1),
                        np.abs(p_diag - P_diag_2),
                        np.abs(p_diag - P_diag_3)
                    ]
                    
                    best_mode_index = np.argmin(errors)
                    
                    if best_mode_index == 0: P_best = p_h 
                    elif best_mode_index == 1: P_best = p_v 
                    else: P_best = (p_h + p_v) // 2 

                    residue[y, x, c] = X_curr - P_best

        if channels == 1:
            residue = residue.reshape(height, width)

    else:
        raise ValueError(f"Modo de predictor '{mode}' no reconocido. Usar 1, 2, 3, 4 o 5.")

    mapped_residue = residue + 32768
    
    # Hacemos clipping al rango de uint16 (0 a 65535)
    np.clip(mapped_residue, 0, 65535, out=mapped_residue)

    return mapped_residue.astype(np.uint16)


def reconstruct_image(residue_array, mode, bits):
    """
    Reconstruye la imagen original a partir de la imagen de residuo.
    ¡Esta operación debe ser un bucle secuencial!

    Args:
        residue_array (np.array): El array con los residuos.
        mode (int): El modo de predicción (1, 2, 3, 4 o 5).
        bits (int): Profundidad de bits de la imagen original (8 o 16).

    Returns:
        np.array: El array de la imagen reconstruida.
    """
    print(f"-> Reconstruyendo desde residuo (modo {mode})...")
    print("-> Desmapeando residuo de rango positivo (int32)...")
    residue_int32 = residue_array.astype(np.int32)
    residue = residue_int32 - 32768
    reconstructed = np.zeros_like(residue, dtype=np.int32)
    
    if len(residue.shape) == 2:
        height, width = residue.shape
        channels = 1
        residue = residue.reshape(height, width, 1)
        reconstructed = reconstructed.reshape(height, width, 1)
    else:
        height, width, channels = residue.shape

    output_dtype = np.uint16 if bits == 16 else np.uint8

    for c in range(channels):
        for y in range(height):
            for x in range(width):
                
                # Píxeles vecinos YA RECONSTRUIDOS
                # a = Izquierda (p_h), b = Arriba (p_v), c = Diagonal (p_diag)
                a = reconstructed[y, x-1, c] if x > 0 else 0
                b = reconstructed[y-1, x, c] if y > 0 else 0
                diag = reconstructed[y-1, x-1, c] if x > 0 and y > 0 else 0
                
                prediction = 0
                
                if mode == 1:
                    prediction = a
                elif mode == 2:
                    prediction = b
                elif mode == 3:
                    prediction = (a + b) // 2
                
                elif mode == 4:
                    # Modo Adaptativo (Conservado)
                    P_diag_1 = reconstructed[y-1, x-2, c] if x > 1 and y > 0 else 0 
                    P_diag_2 = reconstructed[y-2, x-1, c] if y > 1 and x > 0 else 0 
                    P_diag_3 = (P_diag_1 + P_diag_2) // 2

                    errors = [
                        np.abs(diag - P_diag_1),
                        np.abs(diag - P_diag_2),
                        np.abs(diag - P_diag_3)
                    ]
                    best_mode_index = np.argmin(errors)
                    if best_mode_index == 0: prediction = a
                    elif best_mode_index == 1: prediction = b
                    else: prediction = (a + b) // 2
                    
                elif mode == 5:
                    # --- MODO 5: PAETH (Reconstrucción) ---
                    # Lógica idéntica al encoder pero punto a punto
                    p = a + b - diag
                    pa = abs(p - a)
                    pb = abs(p - b)
                    pc = abs(p - diag)
                    
                    if pa <= pb and pa <= pc:
                        prediction = a
                    elif pb <= pc:
                        prediction = b
                    else:
                        prediction = diag
                        
                else:
                    raise ValueError(f"Modo de predictor '{mode}' no reconocido.")
                
                # Reconstruir: X_i = R_i + P_i
                reconstructed[y, x, c] = residue[y, x, c] + prediction
    
    if channels == 1:
        reconstructed = reconstructed.reshape(height, width)

    # Aplicamos clipping por si la reconstrucción diera valores fuera de rango
    max_val = (2**bits) - 1
    np.clip(reconstructed, 0, max_val, out=reconstructed)
    
    return reconstructed.astype(output_dtype)