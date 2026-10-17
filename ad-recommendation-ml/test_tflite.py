import tensorflow as tf
import numpy as np


def run_tflite(model_path, input_data):

    interpreter = tf.lite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nModel:", model_path)

    print(
        "Input shape:",
        input_details[0]["shape"]
    )

    print(
        "Input type:",
        input_details[0]["dtype"]
    )

    print(
        "Output shape:",
        output_details[0]["shape"]
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data.astype(np.float32)
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    return output[0][0]


# ----------------------------------
# Model A
#
# 8 inputs:
#
# 4 interests
# +
# 4 one-hot ad categories
# ----------------------------------

model_a_input = np.array([
    [
        0.90,   # sports interest
        0.20,   # tech interest
        0.10,   # travel interest
        0.30,   # food interest

        1.0,    # sports ad
        0.0,    # tech ad
        0.0,    # travel ad
        0.0     # food ad
    ]
], dtype=np.float32)


prediction_a = run_tflite(
    "models/ad_model_a.tflite",
    model_a_input
)


# ----------------------------------
# Model B
#
# 11 inputs:
#
# Same 8
# +
# region
# hour
# day
# ----------------------------------

model_b_input = np.array([
    [
        0.90,
        0.20,
        0.10,
        0.30,

        1.0,
        0.0,
        0.0,
        0.0,

        0.0,    # region
        19.0,   # 7 PM
        5.0     # Saturday
    ]
], dtype=np.float32)


prediction_b = run_tflite(
    "models/ad_model_b.tflite",
    model_b_input
)


print("\n-----------------------------")
print("TFLite predictions")
print("-----------------------------")

print(
    f"Model A: {prediction_a * 100:.2f}%"
)

print(
    f"Model B: {prediction_b * 100:.2f}%"
)