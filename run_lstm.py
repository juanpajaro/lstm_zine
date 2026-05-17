import keras
import pathlib
from datetime import datetime
from keras import layers
from keras.utils import text_dataset_from_directory


def find_dataset_directories(search_root=None):
    """Encuentra las carpetas imdb_train, imdb_val e imdb_test en la ruta actual."""
    root = pathlib.Path(search_root or pathlib.Path.cwd()).resolve()
    dataset_names = ("imdb_train", "imdb_val", "imdb_test")
    dataset_dirs = {}

    for dataset_name in dataset_names:
        direct_match = root / dataset_name
        if direct_match.is_dir():
            dataset_dirs[dataset_name] = direct_match
            continue

        matches = sorted(path for path in root.rglob(dataset_name) if path.is_dir())
        if not matches:
            raise FileNotFoundError(
                f"No se encontró la carpeta '{dataset_name}' dentro de {root}"
            )
        dataset_dirs[dataset_name] = matches[0]

    return dataset_dirs["imdb_train"], dataset_dirs["imdb_val"], dataset_dirs["imdb_test"]


def load_datasets_from_directories(batch_size=32):
    """Carga los datasets de train/val/test usando text_dataset_from_directory."""
    train_dir, val_dir, test_dir = find_dataset_directories()
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    print(f"Test dir: {test_dir}")

    train_ds = text_dataset_from_directory(train_dir, batch_size=batch_size)
    val_ds = text_dataset_from_directory(val_dir, batch_size=batch_size)
    test_ds = text_dataset_from_directory(test_dir, batch_size=batch_size)

    return train_ds, val_ds, test_ds


def save_versioned_model(model, output_dir="models"):
    """Guarda el modelo con un nombre versionado dentro de models/."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_path / f"lstm_model_v{version}.keras"
    model.save(model_path)

    return model_path


def main(batch_size=32):
    train_ds, val_ds, test_ds = load_datasets_from_directories(batch_size=batch_size)

    max_length = 600
    max_tokens = 30000
    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        split="whitespace",
        output_mode="int",
        output_sequence_length=max_length,
    )
    train_ds_no_labels = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(train_ds_no_labels)

    sequence_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=8
    )
    sequence_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=8
    )
    sequence_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=8
    )

    hidden_dim = 64
    model = keras.Sequential(
        [
            layers.Embedding(max_tokens, hidden_dim, name="embedding", mask_zero=True),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        restore_best_weights=True,
        patience=2,
    )

    history = model.fit(
        sequence_train_ds,
        validation_data=sequence_val_ds,
        epochs=10,
        callbacks=[early_stopping],
    )
    test_loss, test_acc = model.evaluate(sequence_test_ds)

    model_path = save_versioned_model(model)

    # Guardar las métricas de rendimiento en un archivo de texto
    with open("performance_report.txt", "w") as report_file:
        report_file.write(f"Test Loss: {test_loss}\n")
        report_file.write(f"Test Accuracy: {test_acc}\n")

    print("\n Reporte de rendimiento guardado en 'performance_report.txt'")
    print(f"Modelo guardado en '{model_path}'")

if __name__ == "__main__":
    main()