import bisect
import numpy as np
from collections import Counter

# --- 1. Constantes y Utilidades de Bits (arithmetic_common) ---
# Constantes de Precisión (basadas en uint32_t)
P = 32
MAX_VAL = (1 << P) - 1        # 0xFFFFFFFF
TOP_BIT = 1 << (P - 1)      # 0x80000000 (Bit 31)
SECOND_BIT = 1 << (P - 2)   # 0x40000000 (Bit 30)
MASK_30_BITS = SECOND_BIT - 1 # 0x3FFFFFFF (Bits 0 a 29)

class BitWriter:
    def __init__(self):
        self.buffer, self.cur_byte, self.bit_pos = bytearray(), 0, 0
    def write_bit(self, bit):
        self.cur_byte = (self.cur_byte << 1) | (bit & 1)
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.buffer.append(self.cur_byte); self.cur_byte, self.bit_pos = 0, 0
    def flush(self):
        if self.bit_pos > 0:
            self.cur_byte <<= (8 - self.bit_pos)
            self.buffer.append(self.cur_byte)
    def get_bytes(self):
        return bytes(self.buffer)

class BitReader:
    def __init__(self, buffer_bytes):
        self.buffer, self.byte_pos, self.bit_pos = buffer_bytes, 0, 0
    def read_bit(self):
        if self.byte_pos >= len(self.buffer): return 0
        bit = (self.buffer[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos, self.byte_pos = 0, self.byte_pos + 1
        return bit

# --- 2. Codificador Aritmético (ArithmeticEncoder) ---
class ArithmeticEncoder:    
    def __init__(self, bit_writer):
        self.bit_writer = bit_writer
        self.low, self.high, self.underflow = 0, MAX_VAL, 0

    def encode_symbol(self, sym_low, sym_high, total_symbols):
        range_width = (self.high - self.low) + 1
        self.high = self.low + (range_width * sym_high // total_symbols) - 1
        self.low = self.low + (range_width * sym_low // total_symbols)
        self._renormalize()

    def _renormalize(self):
        while True:
            if (self.high ^ self.low) < TOP_BIT: # Caso E1/E2
                bit = (self.high >> 31) & 1
                self.bit_writer.write_bit(bit)
                opposite_bit = 1 - bit
                for _ in range(self.underflow): self.bit_writer.write_bit(opposite_bit)
                self.underflow = 0
                self.low = (self.low << 1) & MAX_VAL
                self.high = ((self.high << 1) | 1) & MAX_VAL
            elif (self.low & SECOND_BIT) and not (self.high & SECOND_BIT): # Caso E3 (Underflow)
                self.underflow += 1
                self.low = self.low & MASK_30_BITS
                self.high = self.high | SECOND_BIT
                self.low = (self.low << 1) & MAX_VAL
                self.high = ((self.high << 1) | 1) & MAX_VAL
            else:
                break

    def finish(self):
        self.underflow += 1
        bit = 0 if self.low < SECOND_BIT else 1
        self.bit_writer.write_bit(bit)
        opposite_bit = 1 - bit
        for _ in range(self.underflow):
            self.bit_writer.write_bit(opposite_bit)
        self.bit_writer.flush()

def encode(symbols_array, symbol_counts_serializable, total_symbols):
    sym_keys = sorted([int(k) for k in symbol_counts_serializable.keys()])
    cum_freq_map, current_low = {}, 0
    for sym_int in sym_keys:
        count = symbol_counts_serializable[str(sym_int)]
        cum_freq_map[sym_int] = (current_low, current_low + count)
        current_low += count
    
    encoder = ArithmeticEncoder(BitWriter())
    for sym in symbols_array.flatten():
        encoder.encode_symbol(*cum_freq_map[sym], total_symbols)
    encoder.finish()
    return encoder.bit_writer.get_bytes()

# --- 3. Decodificador Aritmético (ArithmeticDecoder) ---
class ArithmeticDecoder:
    def __init__(self, bit_reader):
        self.bit_reader = bit_reader
        self.low, self.high, self.code = 0, MAX_VAL, 0
        for _ in range(32):
            self.code = (self.code << 1) | self.bit_reader.read_bit()
    
    def decode_symbol(self, cum_freq_list, cum_freq_map, total_symbols):
        range_width = (self.high - self.low) + 1
        value = ((self.code - self.low + 1) * total_symbols - 1) // range_width
        
        # Búsqueda binaria para encontrar el símbolo
        idx = bisect.bisect_right(cum_freq_list, (value, float('inf')))
        symbol = cum_freq_list[idx-1][1]
        
        # Actualizar el rango
        sym_low, sym_high = cum_freq_map[symbol]
        self.high = self.low + (range_width * sym_high // total_symbols) - 1
        self.low = self.low + (range_width * sym_low // total_symbols)
        
        self._renormalize()
        return symbol

    def _renormalize(self):
        while True:
            if (self.high ^ self.low) < TOP_BIT: # Caso E1/E2
                self.low = (self.low << 1) & MAX_VAL
                self.high = ((self.high << 1) | 1) & MAX_VAL
                self.code = ((self.code << 1) | self.bit_reader.read_bit()) & MAX_VAL
            elif (self.low & SECOND_BIT) and not (self.high & SECOND_BIT): # Caso E3 (Underflow)
                self.low = self.low & MASK_30_BITS
                self.high = self.high | SECOND_BIT
                self.low = (self.low << 1) & MAX_VAL
                self.high = ((self.high << 1) | 1) & MAX_VAL
                self.code = ((self.code ^ SECOND_BIT) << 1 | self.bit_reader.read_bit()) & MAX_VAL
            else:
                break

def decode(bitstream_bytes, frequencies_serializable, total_symbols):
    sym_keys = sorted([int(k) for k in frequencies_serializable.keys()])
    cum_freq_map, cum_freq_list, current_low = {}, [], 0
    for sym_int in sym_keys:
        count = frequencies_serializable[str(sym_int)]
        cum_freq_map[sym_int] = (current_low, current_low + count)
        cum_freq_list.append((current_low, sym_int))
        current_low += count
    
    decoder = ArithmeticDecoder(BitReader(bitstream_bytes))
    decoded_symbols = []
    for _ in range(total_symbols):
        symbol = decoder.decode_symbol(cum_freq_list, cum_freq_map, total_symbols)
        decoded_symbols.append(symbol)
            
    return np.array(decoded_symbols, dtype=np.uint16)