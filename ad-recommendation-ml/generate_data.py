import numpy as np
import pandas as pd

# Makes the generated data reproducible.
np.random.seed(42)

NUM_ROWS = 200_000

# -------------------------------
# 1. User behavior / interests
# -------------------------------

sports_interest = np.random.random(NUM_ROWS)
tech_interest = np.random.random(NUM_ROWS)
travel_interest = np.random.random(NUM_ROWS)
food_interest = np.random.random(NUM_ROWS)

# -------------------------------
# 2. Advertisement category
#
# 0 = Sports
# 1 = Technology
# 2 = Travel
# 3 = Food
# -------------------------------

ad_category = np.random.randint(
    0,
    4,
    NUM_ROWS
)

# -------------------------------
# 3. Context
# -------------------------------

# Artificial coarse regions:
# 0, 1, 2, 3
region = np.random.randint(
    0,
    4,
    NUM_ROWS
)

# Hour: 0-23
hour = np.random.randint(
    0,
    24,
    NUM_ROWS
)

# Monday=0 ... Sunday=6
day_of_week = np.random.randint(
    0,
    7,
    NUM_ROWS
)

# -------------------------------
# 4. Find interest matching
#    the advertisement
# -------------------------------

interests = np.column_stack([
    sports_interest,
    tech_interest,
    travel_interest,
    food_interest
])

interest_match = interests[
    np.arange(NUM_ROWS),
    ad_category
]

# -------------------------------
# 5. Context signals
# -------------------------------

# Evening users are somewhat more
# likely to interact in our simulated world.
evening = (
    (hour >= 17) &
    (hour <= 22)
).astype(float)

# Artificial example:
# some region/category combinations
# perform better.
region_match = (
    region == ad_category
).astype(float)

# Weekend signal
weekend = (
    day_of_week >= 5
).astype(float)

# -------------------------------
# 6. Hidden "real-world" rule
# -------------------------------

noise = np.random.normal(
    0,
    0.5,
    NUM_ROWS
)

score = (
    -4.0
    + 3.0 * interest_match
    + 0.50 * evening
    + 0.70 * region_match
    + 0.30 * weekend
    + noise
)

# -------------------------------
# 7. Convert score to probability
# -------------------------------

click_probability = (
    1 /
    (1 + np.exp(-score))
)

# -------------------------------
# 8. Generate click/no-click
# -------------------------------

clicked = np.random.binomial(
    1,
    click_probability
)

# -------------------------------
# 9. Create DataFrame
# -------------------------------

df = pd.DataFrame({

    "sports_interest": sports_interest,
    "tech_interest": tech_interest,
    "travel_interest": travel_interest,
    "food_interest": food_interest,

    "ad_category": ad_category,

    "region": region,
    "hour": hour,
    "day_of_week": day_of_week,

    "clicked": clicked
})

# -------------------------------
# 10. Save dataset
# -------------------------------

df.to_csv(
    "data/ad_impressions.csv",
    index=False
)

print(df.head())

print("\nRows:", len(df))

print(
    "Overall CTR:",
    df["clicked"].mean()
)