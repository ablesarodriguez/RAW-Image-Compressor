import argparse
import numpy as np
from src import io
from src import entropy
from src import processing
from src import quantization
from src import metrics
from src import predictor
from src import codec

def main():
    parser = argparse.ArgumentParser(description="Herramienta de Compresión de Imágenes RAW")
    subparsers = parser.add_subparsers(dest='operation', required=True, help="Operación a realizar")

    # --- Argumentos Comunes ---
    def add_common_args(p):
        p.add_argument('--width', required=True, type=int)
        p.add_argument('--height', required=True, type=int)
        p.add_argument('--channels', required=True, type=int)
        p.add_argument('--encoding', required=True, type=str)

    # --- Comandos Existentes (Read, Write, Entropy, etc) ---
    p_read = subparsers.add_parser('read', help="Lee información de imagen")
    p_read.add_argument('inputpath', type=str)
    add_common_args(p_read)

    p_write = subparsers.add_parser('write', help="Procesa y guarda imagen")
    p_write.add_argument('inputpath', type=str)
    p_write.add_argument('outputpath', type=str)
    add_common_args(p_write)
    p_write.add_argument('--output-encoding', type=str)

    p_entropy = subparsers.add_parser('entropy', help="Calcula entropía")
    p_entropy.add_argument('inputpath', type=str)
    add_common_args(p_entropy)

    p_quant = subparsers.add_parser('quantize', help="Cuantiza imagen")
    p_quant.add_argument('inputpath', type=str)
    p_quant.add_argument('outputpath', type=str)
    add_common_args(p_quant)
    p_quant.add_argument('--qstep', required=True, type=int)

    p_dequant = subparsers.add_parser('dequantize', help="Decuantiza imagen")
    p_dequant.add_argument('inputpath', type=str)
    p_dequant.add_argument('outputpath', type=str)
    add_common_args(p_dequant)
    p_dequant.add_argument('--qstep', required=True, type=int)
    p_dequant.add_argument('--original', type=str)
    p_dequant.add_argument('--original-encoding', type=str)

    p_pred = subparsers.add_parser('predict', help="Genera residuo")
    p_pred.add_argument('inputpath', type=str)
    p_pred.add_argument('outputpath', type=str)
    add_common_args(p_pred)
    p_pred.add_argument('--mode', required=True, type=int, choices=[1, 2, 3, 4, 5])

    p_rec = subparsers.add_parser('reconstruct', help="Reconstruye desde residuo")
    p_rec.add_argument('inputpath', type=str)
    p_rec.add_argument('outputpath', type=str)
    add_common_args(p_rec)
    p_rec.add_argument('--mode', required=True, type=int, choices=[1, 2, 3, 4, 5])
    p_rec.add_argument('--original', type=str)
    p_rec.add_argument('--original-encoding', type=str)

    p_enc = subparsers.add_parser('encode', help="Comprime imagen completa (.comp)")
    p_enc.add_argument('inputpath', type=str)
    p_enc.add_argument('outputpath', type=str)
    add_common_args(p_enc)
    p_enc.add_argument('--qstep', required=True, type=int)
    p_enc.add_argument('--mode', required=True, type=int, choices=[1, 2, 3, 4, 5])

    p_dec = subparsers.add_parser('decode', help="Descomprime archivo (.comp)")
    p_dec.add_argument('inputpath', type=str)
    p_dec.add_argument('outputpath', type=str)

    args = parser.parse_args()

    # --- Ejecución ---
    
    # Operaciones que NO cargan imagen RAW inicialmente
    if args.operation == 'decode':
        codec.decode_image(args.inputpath, args.outputpath)
        return

    if args.operation == 'encode':
        codec.encode_image(
            args.inputpath, args.outputpath, 
            args.width, args.height, args.channels, args.encoding,
            args.qstep, args.mode
        )
        return

    # Operaciones comunes que cargan la imagen RAW
    print(f"\nCargando imagen: '{args.inputpath}'")
    image_data = io.load_raw_image(
        args.inputpath, args.width, args.height, args.channels, args.encoding
    )
    if not image_data: return

    if args.operation == 'read':
        p = image_data['params']
        print(f"\nInfo: {p['width']}x{p['height']}x{p['channels']} | {p['bits']} bits | {p['encoding']}")

    elif args.operation == 'write':
        out_enc = args.output_encoding if args.output_encoding else args.encoding
        final = processing.convert_encoding(image_data['array'], args.encoding, out_enc)
        io.save_raw_image(final, args.outputpath)

    elif args.operation == 'entropy':
        e0, e1 = entropy.calculate_entropy(image_data['array'])
        print(f"Entropía orden 0: {e0:.4f} | Orden 1: {e1:.4f}")

    elif args.operation == 'quantize':
        q = quantization.quantize_image(image_data['array'], args.qstep)
        io.save_raw_image(q, args.outputpath)

    elif args.operation == 'dequantize':
        dq = quantization.dequantize_image(image_data['array'], args.qstep)
        io.save_raw_image(dq, args.outputpath)
        if args.original:
            orig_data = io.load_raw_image(args.original, args.width, args.height, args.channels, args.original_encoding)
            if orig_data:
                m = metrics.calculate_metrics(orig_data['array'], dq, orig_data['params']['bits'])
                print(f"Metrics -> PAE: {m['PAE']}, MSE: {m['MSE']:.4f}, PSNR: {m['PSNR']:.2f} dB")

    elif args.operation == 'predict':
        res = predictor.predict_image(image_data['array'], args.mode)
        io.save_raw_image(res, args.outputpath)

    elif args.operation == 'reconstruct':
        orig_bits = 16
        if args.original:
            orig_data = io.load_raw_image(args.original, args.width, args.height, args.channels, args.original_encoding)
            if orig_data: orig_bits = orig_data['params']['bits']
        
        rec = predictor.reconstruct_image(image_data['array'], args.mode, orig_bits)

        out_enc = args.original_encoding if args.original_encoding else f"ube{orig_bits}"
        final = processing.convert_encoding(rec, f"ule{orig_bits}", out_enc)
        io.save_raw_image(final, args.outputpath)
        
        if args.original and orig_data:
            m = metrics.calculate_metrics(orig_data['array'], rec, orig_bits)
            print(f"Metrics -> PSNR: {m['PSNR']:.2f} dB")

if __name__ == "__main__":
    main()