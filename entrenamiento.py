#!/usr/bin/env python3
"""
Script de entrenamiento para modelo LSTM de análisis de sentimientos.
Descarga el dataset IMDB, lo procesa y prepara los datos para entrenamiento.
"""

import argparse
import os
import pathlib
import random
import shutil

import keras
from keras import layers
from keras.utils import text_dataset_from_directory
import matplotlib.pyplot as plt


def main(batch_size=32, val_percentage=0.2):
    """
    Función principal que ejecuta el pipeline de preparación de datos.
    
    Args:
        batch_size (int): Tamaño de batch para los datasets. Por defecto 32.
        val_percentage (float): Porcentaje de datos para validación. Por defecto 0.2.
    """
    print("Iniciando descarga del dataset IMDB...")
    zip_path = keras.utils.get_file(
        origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        fname="imdb",
        extract=True,
    )

    imdb_extract_dir = pathlib.Path(zip_path) / "aclImdb"

    print("Directorios encontrados:")
    for path in imdb_extract_dir.glob("*/*"):
        if path.is_dir():
            print(path)
    
    print("\nEjemplo de reseña:")
    print(open(imdb_extract_dir / "train" / "pos" / "4077_10.txt", "r").read())

    train_dir = pathlib.Path("imdb_train")
    test_dir = pathlib.Path("imdb_test")
    val_dir = pathlib.Path("imdb_val")

    print("\nCopiando datos de prueba...")
    shutil.copytree(imdb_extract_dir / "test", test_dir)

    print(f"Dividiendo datos con {val_percentage*100}% para validación...")
    for category in ("neg", "pos"):
        src_dir = imdb_extract_dir / "train" / category
        src_files = os.listdir(src_dir)
        random.Random(1337).shuffle(src_files)
        num_val_samples = int(len(src_files) * val_percentage)

        os.makedirs(val_dir / category)
        for file in src_files[:num_val_samples]:
            shutil.copy(src_dir / file, val_dir / category / file)
        os.makedirs(train_dir / category)
        for file in src_files[num_val_samples:]:
            shutil.copy(src_dir / file, train_dir / category / file)

    print("Creando datasets...")
    train_ds = text_dataset_from_directory(train_dir, batch_size=batch_size)
    val_ds = text_dataset_from_directory(val_dir, batch_size=batch_size)
    test_ds = text_dataset_from_directory(test_dir, batch_size=batch_size)
    
    print(f"Datasets creados exitosamente con batch_size={batch_size}")
    
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara el dataset IMDB para entrenamiento de modelo LSTM"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para los datasets (default: 32)"
    )
    parser.add_argument(
        "--val-percentage",
        type=float,
        default=0.2,
        help="Porcentaje de datos para validación (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    try:
        train_ds, val_ds, test_ds = main(
            batch_size=args.batch_size,
            val_percentage=args.val_percentage
        )
        print("\n✓ Script completado exitosamente")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)