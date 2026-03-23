import os
import subprocess
import csv
import sys
from src import io       
from src import metrics  

# --- CONFIGURACIÓN ---
IMAGES_TO_PROCESS = [
    {
        "name": "n1_GRAY",
        "path": "./images/n1_GRAY.ube8_1_2560_2048.raw",
        "width": 2560, "height": 2048, "channels": 1,
        "encoding": "ube8", "bits": 8
    }
]

# --- RUTAS DE SALIDA ---
OUTPUT_DIR = "./out_benchmark"
TEMP_COMP = os.path.join(OUTPUT_DIR, "temp_benchmark.comp")
TEMP_REC = os.path.join(OUTPUT_DIR, "temp_benchmark_rec.raw")

def run_benchmark():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    for img in IMAGES_TO_PROCESS:
        print(f"\nProcesando: {img['name']}")
        
        if not os.path.exists(img['path']):
            print(f"  [ERROR] No se encuentra: {img['path']}")
            continue

        csv_filename = os.path.join(OUTPUT_DIR, f"{img['name']}_metrics.csv")
        
        # Cargar original para métricas
        original_data = io.load_raw_image(img['path'], img['width'], img['height'], img['channels'], img['encoding'])
        if original_data is None: continue
        original_array = original_data['array']

        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(['qstep', 'bps', 'PSNR', 'MSE'])

            for q in range(1, 3): # Modificar para analizar más qstep
                # 1. ENCODE (Argumentos posicionales: input output)
                cmd_enc = [
                    "python", "main.py", "encode",
                    img['path'], TEMP_COMP,
                    "--width", str(img['width']), "--height", str(img['height']),
                    "--channels", str(img['channels']), "--encoding", img['encoding'],
                    "--qstep", str(q), "--mode", "5"
                ]
                subprocess.run(cmd_enc, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 2. BPS
                file_size = os.path.getsize(TEMP_COMP)
                bps = (file_size * 8) / (img['width'] * img['height'] * img['channels'])

                # 3. DECODE (Argumentos posicionales: input output)
                cmd_dec = ["python", "main.py", "decode", TEMP_COMP, TEMP_REC]
                subprocess.run(cmd_dec, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 4. MÉTRICAS
                rec_data = io.load_raw_image(TEMP_REC, img['width'], img['height'], img['channels'], "sle16")
                if rec_data:
                    m = metrics.calculate_metrics(original_array, rec_data['array'], img['bits'])
                    writer.writerow([q, f"{bps:.4f}", f"{m['PSNR']:.4f}", f"{m['MSE']:.4f}"])
                    sys.stdout.write(f"\r  -> qstep: {q:03d} | bps: {bps:.4f} | PSNR: {m['PSNR']:.2f} | MSE: {m['MSE']:.4f}")
                    sys.stdout.flush()

    # Limpieza
    if os.path.exists(TEMP_COMP): os.remove(TEMP_COMP)
    if os.path.exists(TEMP_REC): os.remove(TEMP_REC)
    print(f"\nBenchmark completado. Resultados en '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_benchmark()