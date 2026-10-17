import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/model_a.keras"
)

# Same user's interests
sports = 0.90
tech = 0.20
travel = 0.10
food = 0.30

ads = {
    "Sports": 0,
    "Technology": 1,
    "Travel": 2,
    "Food": 3
}

results = []

for ad_name, ad_category in ads.items():

    features = np.array([
        [
            sports,
            tech,
            travel,
            food,
            float(ad_category)
        ]
    ], dtype=np.float32)

    prediction = model.predict(
        features,
        verbose=0
    )[0][0]

    results.append(
        (ad_name, prediction)
    )

# Highest probability first
results.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nAd Ranking")
print("----------------------")

for ad, probability in results:
    print(
        f"{ad:12} "
        f"{probability * 100:.2f}%"
    )

print(
    "\nRecommended Ad:",
    results[0][0]
)