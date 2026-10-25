"""Custom CNN architecture matching the layer table in the paper."""

from pathlib import Path

from tensorflow import keras

CLASS_NAMES = ("Normal", "Benign", "Malignant")
INPUT_SHAPE = (256, 256, 1)


def build_model(compile_model: bool = True) -> keras.Model:
    """Build the exact two-convolution Keras model described in the paper.

    Valid 3x3 convolutions and 2x2/stride-2 pooling reproduce the reported
    shapes and parameter counts: 254, 127, 125, 62, 246016, 3936272.
    Missing activations/units are resolved as ReLU and Dense(16, ReLU).
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=INPUT_SHAPE, name="ct_slice"),
            keras.layers.Conv2D(64, (3, 3), padding="valid", name="conv2d"),
            keras.layers.Activation("relu", name="activation"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"),
            keras.layers.Conv2D(64, (3, 3), padding="valid", activation="relu", name="conv2d_1"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(16, activation="relu", name="dense"),
            keras.layers.Dense(3, activation="softmax", name="class_probabilities"),
        ],
        name="lung_ct_custom_cnn",
    )
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def load_model(path: str | Path) -> keras.Model:
    """Load and validate a trained Keras checkpoint."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. Train a model first; "
            "the research paper does not provide checkpoint weights."
        )
    if checkpoint.name.endswith(".weights.h5"):
        model = build_model(compile_model=False)
        model.load_weights(checkpoint)
    else:
        model = keras.models.load_model(checkpoint)
    if tuple(model.input_shape[1:]) != INPUT_SHAPE or model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Checkpoint input/output shapes do not match the published model")
    return model
