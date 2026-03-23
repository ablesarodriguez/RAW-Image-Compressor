# RAW Image Compression System

Este proyecto implementa un sistema completo de compresión y descompresión de imágenes RAW (sin procesar) utilizando Python. El objetivo es explorar y aplicar técnicas fundamentales de compresión de imagen, desde la cuantización hasta la codificación aritmética, pasando por la predicción espacial y el análisis de métricas de distorsión.

El sistema es capaz de procesar imágenes con diferentes profundidades de bits (8 y 16 bits), múltiples canales (escala de grises, RGB, hiperespectrales) y distinta *endianness*.

## Estructura del Proyecto

El código sigue una arquitectura modular organizada de la siguiente manera:

* **`src/`**: Paquete principal con la lógica del compresor.
    * `io.py`: Gestión de lectura/escritura RAW.
    * `processing.py`: Normalización y conversión de formatos.
    * `quantization.py`: Cuantización uniforme y decuantización.
    * `predictor.py`: Predictores espaciales (Modos 1-5, incluyendo Paeth).
    * `entropy.py` & `metrics.py`: Cálculo de entropía, MSE y PSNR.
    * `arithmetic.py`: Motor de codificación aritmética binaria.
    * `codec.py`: Orquestador de los procesos de *encode* y *decode*.
* **`main.py`**: Interfaz de línea de comandos (CLI) principal para ejecutar operaciones individuales (lectura, entropía, cuantización) o procesos completos (compresión/descompresión).
* **`benchmark.py`**: Script automatizado para generar comparativas de rendimiento (Curvas Rate-Distortion).
* **`commands/`**: Contiene archivos de texto con los comandos exactos necesarios para procesar las imágenes de muestra incluidas.
* **`images/`**: Banco de imágenes RAW para pruebas.
* **`out/`** y **`out_benchmark/`**: Directorios donde se generan los archivos resultantes de las operaciones.

## Fases de Desarrollo

1.  **Lectura y Escritura (I/O)**: Implementación de parsers para interpretar formatos RAW (`uint8`, `int16`, `uint16`) y gestión de *endianness* (Little/Big Endian).
2.  **Cuantización Básica**: Desarrollo de un cuantizador uniforme escalar y su inversa.
3.  **Análisis de Información**: Implementación de herramientas para medir Entropía (Orden 0 y 1) y métricas de error (PAE, MSE, PSNR).
4.  **Predicción Espacial**: Implementación de predictores lineales (Horizontal, Vertical, Average) para descorrelacionar espacialmente los datos.
5.  **Codificación Entrópica**: Integración de un Codificador Aritmético para generar el *bitstream* final comprimido (`.comp`) con cabeceras JSON.
6.  **Mejoras y Benchmarking**:
    * Creación de un sistema de *benchmark* para validar mejoras objetivamente.
    * Refinamiento de la cuantización (redondeo aritmético y clipping).
    * Implementación de predictores avanzados: **Modo 4** (Adaptativo) y **Modo 5** (Paeth, estándar PNG).

## Requisitos

* **Python 3.8+**
* **Librerías Python**:
    * `numpy`: Para manipulación eficiente de matrices y cálculos matemáticos.
    
    ```bash
    pip install numpy
    ```

## Uso y Ejecución

### Operaciones Principales (`main.py`)
Script central. Permite leer imágenes, calcular entropías, comprimir (`encode`) y descomprimir (`decode`).

> **Nota:** Comandos exactos de ejecución ubicados en la carpeta **`commands/`**.

Ejemplo de ejecución:
1.  Consultar `commands/07_n1_gray.txt`.
2.  Copiar y ejecutar el comando de compresión (`encode`).
3.  Copiar y ejecutar el comando de descompresión (`decode`).

### Comparativas (`benchmark.py`)
Para analizar el rendimiento del compresor:
1.  Ejecuta `python benchmark.py`.
2.  El script procesará las imágenes automáticamente variando el paso de cuantización (`qstep`).
3.  Los resultados se guardan en un archivo `.csv`.

## Visualización de Imágenes (Fiji)

Dado que las imágenes utilizadas son **RAW** (datos puros sin cabecera estándar como BMP o JPG), no se pueden visualizar con visores de imágenes convencionales.

Para visualizar tanto las imágenes originales (`images/`) como las reconstruidas (`out/`), utilizamos **Fiji**:

1.  Abrir Fiji.
2.  Ir a `File` -> `Import` -> `Raw...`
3.  Seleccionar el archivo `.raw`.
4.  Introducir los parámetros de la imagen.

##