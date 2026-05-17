import keras
from keras import layers

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

def main():
    history = model.fit(
        sequence_train_ds,
        validation_data=sequence_val_ds,
        epochs=10,
        callbacks=[early_stopping],
    )
    test_loss, test_acc = model.evaluate(sequence_test_ds)

    # Guardar las métricas de rendimiento en un archivo de texto
    with open("performance_report.txt", "w") as report_file:
        report_file.write(f"Test Loss: {test_loss}\n")
        report_file.write(f"Test Accuracy: {test_acc}\n")

    print("\n✓ Reporte de rendimiento guardado en 'performance_report.txt'")

if __name__ == "__main__":
    main()