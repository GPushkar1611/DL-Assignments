import numpy as np
import pandas as pd

np.random.seed(42)

# ----------------------------
# Parameters
# ----------------------------
N_BASE = 500
N_TOTAL = 5000

# ----------------------------
# Base dataset (500 samples)
# ----------------------------
data = []

for _ in range(N_BASE):
    day_of_week = np.random.randint(0, 7)
    meal_time = np.random.randint(0, 3)

    is_weekend = int(day_of_week >= 5)

    row = {
        "student_id": np.random.randint(1, 201),
        "day_of_week": day_of_week,
        "meal_time": meal_time,
        "meal_type": np.random.randint(0, 2),
        "is_weekend": is_weekend,
        "class_day": np.random.randint(0, 2),
        "assignment_deadline": np.random.randint(0, 2),
        "temperature": np.random.normal(30, 3),
        "humidity": np.random.normal(60, 10),
        "wind_speed": np.random.normal(5, 1),
        "rain": np.random.randint(0, 2),
        "air_quality": np.random.normal(100, 20),
        "rising_time": np.random.randint(300, 600),
        "sleeping_time": np.random.randint(1320, 1440),
    }

    # Simple, transparent target
    row["mess_duration"] = max(
        5,
        20
        + 5 * meal_time
        + 10 * is_weekend
        - 5 * row["assignment_deadline"]
        + np.random.normal(0, 5)
    )

    data.append(row)

df = pd.DataFrame(data)

# ----------------------------
# Augmentation to 5000 samples
# ----------------------------
augmented = []

while len(df) + len(augmented) < N_TOTAL:
    sample = df.sample(1).iloc[0].copy()

    # Add Gaussian noise ONLY where asked
    sample["temperature"] += np.random.normal(0, 1)
    sample["humidity"] += np.random.normal(0, 3)
    sample["wind_speed"] += np.random.normal(0, 0.5)
    sample["air_quality"] += np.random.normal(0, 5)
    sample["rising_time"] += np.random.normal(0, 10)
    sample["sleeping_time"] += np.random.normal(0, 10)

    augmented.append(sample)

df_final = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)

# ----------------------------
# Save dataset
# ----------------------------
df_final.to_csv("iit_h_mess_dataset.csv", index=False)
print("Dataset saved as iit_h_mess_dataset.csv")