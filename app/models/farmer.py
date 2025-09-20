import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========== Load Dataset ==========
data = {
    "pH": [3.716080075, 8.099124189, 8.316765884],
    "TDS": [18630.05786, 19909.54173, 22018.41744],
    "Hardness": [129.4229205, 224.2362594, 214.3733941],
    "Turbidity": [4.500656275, 3.05593375, 4.628770537]
}

df = pd.DataFrame(data)

# ========== Create Labels (Beneficial / Not) ==========
def label_water(row):
    if 6.5 <= row["pH"] <= 8.4 and row["TDS"] < 500 and row["Hardness"] < 300 and row["Turbidity"] < 5:
        return 1  # Beneficial
    else:
        return 0  # Not Beneficial

df["Beneficial"] = df.apply(label_water, axis=1)

# ========== Train ML Model ==========
X = df[["pH", "TDS", "Hardness", "Turbidity"]]
y = df["Beneficial"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=35)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

if len(y_test) > 0:  # to avoid errors with very small dataset
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

# ========== IoT Decision Pipeline ==========
def iot_reading():
    """
    Simulate a reading from IoT sensors. Replace with actual sensor values.
    """
    # Example values (replace with real IoT input)
    pH_val = 7.2
    TDS_val = 1200
    hardness_val = 180
    turbidity_val = 3.5
    return [pH_val, TDS_val, hardness_val, turbidity_val]

def decision_pipeline():
    sensor = iot_reading()
    pred = model.predict([sensor])[0]
    print(f"Sensor readings → pH: {sensor[0]}, TDS: {sensor[1]}, Hardness: {sensor[2]}, Turbidity: {sensor[3]}")

    if pred == 1:
        print("✅ Water is Beneficial for Crops. Inform farmer.")
    else:
        print("⚠️ Water is NOT Beneficial. Trigger mechanism to adjust water quality.")

if __name__ == "__main__":
    decision_pipeline()
