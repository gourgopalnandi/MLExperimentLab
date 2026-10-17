import tensorflow as tf
import os
import shutil


def convert_model(
    keras_path,
    saved_model_path,
    tflite_path
):
    print(f"\nLoading: {keras_path}")

    model = tf.keras.models.load_model(
        keras_path
    )

    # Remove previous SavedModel directory
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)

    print("Exporting SavedModel...")

    model.export(
        saved_model_path
    )

    print("Converting SavedModel to TFLite...")

    converter = (
        tf.lite.TFLiteConverter
        .from_saved_model(saved_model_path)
    )

    tflite_model = converter.convert()

    with open(
        tflite_path,
        "wb"
    ) as f:
        f.write(tflite_model)

    print(f"Created: {tflite_path}")
    print(
        f"Size: {len(tflite_model) / 1024:.2f} KB"
    )


convert_model(
    "models/ad_model_a.keras",
    "models/saved_model_a",
    "models/ad_model_a.tflite"
)

convert_model(
    "models/ad_model_b.keras",
    "models/saved_model_b",
    "models/ad_model_b.tflite"
)

print("\nDone!")