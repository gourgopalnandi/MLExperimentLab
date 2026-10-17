import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


# ----------------------------------
# 1. Load our generated dataset
# ----------------------------------

df = pd.read_csv(
    "data/ad_impressions.csv"
)

print("Dataset size:", len(df))
print("Overall CTR:", df["clicked"].mean())


# ----------------------------------
# 2. Features available to Model A
# ----------------------------------

FEATURES_A = [
    "sports_interest",
    "tech_interest",
    "travel_interest",
    "food_interest",
    "ad_category"
]

X = df[FEATURES_A].astype("float32")

# This is what we want to predict.
y = df["clicked"].astype("float32")


# ----------------------------------
# 3. Split into train / validation / test
# ----------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))


# ----------------------------------
# 4. Build Model A
# ----------------------------------

model_a = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(len(FEATURES_A),)
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
])


# ----------------------------------
# 5. Compile
# ----------------------------------

model_a.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ----------------------------------
# 6. Show architecture
# ----------------------------------

model_a.summary()


# ----------------------------------
# 7. Train
# ----------------------------------

history = model_a.fit(

    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=10,

    batch_size=256
)


# ----------------------------------
# 8. Evaluate using unseen test data
# ----------------------------------

test_loss, test_auc = model_a.evaluate(
    X_test,
    y_test
)

print("\nModel A Results")
print("----------------")
print("Test Loss:", test_loss)
print("Test AUC:", test_auc)


# ----------------------------------
# 9. Save trained model
# ----------------------------------

model_a.save(
    "models/model_a.keras"
)

print(
    "\nSaved Model A to models/model_a.keras"
)