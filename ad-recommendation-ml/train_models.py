import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATA_FILE = "data/ad_impressions.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)


# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("Dataset rows:", len(df))
print(f"Overall CTR: {df['clicked'].mean() * 100:.2f}%")


# -------------------------------------------------
# 2. One-hot encode ad category
#
# Original:
# 0 = Sports
# 1 = Technology
# 2 = Travel
# 3 = Food
#
# Instead of:
#
# ad_category = 2
#
# we create:
#
# ad_sports = 0
# ad_tech   = 0
# ad_travel = 1
# ad_food   = 0
# -------------------------------------------------

ad_one_hot = pd.get_dummies(
    df["ad_category"],
    prefix="ad",
    dtype="float32"
)

# Rename for readability
ad_one_hot.columns = [
    "ad_sports",
    "ad_tech",
    "ad_travel",
    "ad_food"
]

df = pd.concat(
    [
        df.drop(columns=["ad_category"]),
        ad_one_hot
    ],
    axis=1
)


# -------------------------------------------------
# 3. Model A features
# -------------------------------------------------

FEATURES_A = [
    "sports_interest",
    "tech_interest",
    "travel_interest",
    "food_interest",

    "ad_sports",
    "ad_tech",
    "ad_travel",
    "ad_food"
]


# -------------------------------------------------
# 4. Model B features
#
# Model B gets everything Model A gets,
# plus contextual features.
# -------------------------------------------------

FEATURES_B = FEATURES_A + [
    "region",
    "hour",
    "day_of_week"
]


# -------------------------------------------------
# 5. Split ROW INDICES
#
# This is important:
#
# Model A and B must use exactly the same
# train/validation/test impressions.
# -------------------------------------------------

indices = np.arange(len(df))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=df["clicked"]
)

validation_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=df.iloc[temp_indices]["clicked"]
)


print("\nDataset split")
print("------------------------")
print("Training:", len(train_indices))
print("Validation:", len(validation_indices))
print("Test:", len(test_indices))


# -------------------------------------------------
# 6. Helper function
# -------------------------------------------------

def get_data(features):

    X = df[features].astype("float32").values
    y = df["clicked"].astype("float32").values

    return (
        X[train_indices],
        X[validation_indices],
        X[test_indices],
        y[train_indices],
        y[validation_indices],
        y[test_indices]
    )


(
    X_train_a,
    X_val_a,
    X_test_a,
    y_train,
    y_val,
    y_test
) = get_data(FEATURES_A)


(
    X_train_b,
    X_val_b,
    X_test_b,
    _,
    _,
    _
) = get_data(FEATURES_B)


# -------------------------------------------------
# 7. Build neural network
# -------------------------------------------------

def create_model(input_size, name):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(input_size,)
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid"
            )
        ],
        name=name
    )

    model.compile(
        optimizer="adam",

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model


# -------------------------------------------------
# 8. Create Model A
# -------------------------------------------------

model_a = create_model(
    len(FEATURES_A),
    "model_a"
)


# -------------------------------------------------
# 9. Create Model B
# -------------------------------------------------

model_b = create_model(
    len(FEATURES_B),
    "model_b"
)


print("\nMODEL A")
model_a.summary()

print("\nMODEL B")
model_b.summary()


# -------------------------------------------------
# 10. Train Model A
# -------------------------------------------------

print("\n")
print("=" * 60)
print("TRAINING MODEL A")
print("=" * 60)

model_a.fit(
    X_train_a,
    y_train,

    validation_data=(
        X_val_a,
        y_val
    ),

    epochs=10,
    batch_size=256,
    verbose=1
)


# -------------------------------------------------
# 11. Train Model B
# -------------------------------------------------

print("\n")
print("=" * 60)
print("TRAINING MODEL B")
print("=" * 60)

model_b.fit(
    X_train_b,
    y_train,

    validation_data=(
        X_val_b,
        y_val
    ),

    epochs=10,
    batch_size=256,
    verbose=1
)


# -------------------------------------------------
# 12. Predictions on SAME test impressions
# -------------------------------------------------

pred_a = model_a.predict(
    X_test_a,
    verbose=0
).flatten()

pred_b = model_b.predict(
    X_test_b,
    verbose=0
).flatten()


# -------------------------------------------------
# 13. Evaluate
# -------------------------------------------------

auc_a = roc_auc_score(
    y_test,
    pred_a
)

auc_b = roc_auc_score(
    y_test,
    pred_b
)

logloss_a = log_loss(
    y_test,
    pred_a
)

logloss_b = log_loss(
    y_test,
    pred_b
)


print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Model A AUC      : {auc_a:.4f}"
)

print(
    f"Model B AUC      : {auc_b:.4f}"
)

print()

print(
    f"Model A Log Loss : {logloss_a:.4f}"
)

print(
    f"Model B Log Loss : {logloss_b:.4f}"
)


print("\nAUC Improvement")

auc_improvement = auc_b - auc_a

print(
    f"{auc_improvement:+.4f}"
)


# -------------------------------------------------
# 14. Save models
# -------------------------------------------------

model_a.save(
    "models/ad_model_a.keras"
)

model_b.save(
    "models/ad_model_b.keras"
)

print("\nModels saved:")
print("models/ad_model_a.keras")
print("models/ad_model_b.keras")