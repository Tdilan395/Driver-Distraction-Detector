# Sistema de Detección de Distracciones para Conductores

## Bienvenida

Bienvenido al repositorio del prototipo de sistema de detección de distracciones para conductores. Este proyecto tiene como objetivo implementar y evaluar un sistema que integra hardware y software para identificar distracciones y emitir alertas en tiempo real, aumentando así la seguridad vial.

## Contexto

Este proyecto surge de la necesidad de mejorar la seguridad en la conducción, reduciendo los accidentes causados por distracciones al volante. El prototipo desarrollado se somete a pruebas en entornos controlados, y la retroalimentación de conductores y expertos en seguridad vial es fundamental para su mejora continua.

## Descripción del Script

El script `detect_distractions.py` utiliza tecnologías de visión por computadora y procesamiento de audio para:

- Detectar distracciones en conductores en tiempo real.
- Emitir alertas sonoras específicas según el tipo de distracción detectada.

### Características Principales

- **Visión por Computadora**: Utiliza `OpenCV` para la captura y procesamiento de imágenes.
- **Modelos de Detección YOLO**: Implementa modelos YOLO entrenados para identificar comportamientos distractores.
- **Alertas Sonoras**: Genera y reproduce alertas sonoras utilizando `pygame` y `gtts` (Google Text-to-Speech).
- **Registro de Eventos**: Almacena imágenes de eventos de distracción y registra detalles en `data.csv`.

### Archivos Generados

- **Imágenes de Eventos**: Almacenadas en la carpeta `captures`.
- **Registro de Eventos**: `data.csv` incluye detalles como la hora, la clase de distracción y la confianza.

## Estructura del Proyecto

- `src/detect_distractions.py`: Script principal para la detección de distracciones.
- `requirements.txt`: Dependencias necesarias.
- `resources/`: Contiene modelos YOLO y archivos de audio para alertas.
- `src/captures/`: Imágenes capturadas durante eventos de distracción.
- `src/data.csv`: Detalles de cada distracción detectada, incluyendo nombre de imagen, tipo, confianza y marca de tiempo.
- `src/distracciones.png`: Gráfico de barras de las distracciones detectadas.

## Instalación y Ejecución

### Configuración del Entorno

### Configuración del Entorno

1. **Crear y activar un entorno virtual**:
   > py -m venv ./

   > Scripts/activate

2. **Instalar las dependencias**:
   > pip install -r requirements.txt

3. **Ejecución del Script para iniciar el sistema de detección de distracciones:**:
   > cd src

   > py detect_distractions.py
