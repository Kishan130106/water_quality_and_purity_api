import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------
# Step 1: Load Dataset
# -----------------------
data = {
    "pH": [3.716080075, 8.099124189, 8.316765884],
    "TDS": [18630.05786, 19909.54173, 22018.41744],
    "Hardness": [129.4229205, 224.2362594, 214.3733941],
    "Turbidity": [4.500656275, 3.05593375, 4.628770537]
}
df = pd.DataFrame(data)

# -----------------------
# Step 2: Generate Labels
# -----------------------
def label_quality(row):
    if (6.5 <= row["pH"] <= 8.5) and (row["TDS"] < 500) and (row["Hardness"] < 300) and (row["Turbidity"] < 5):
        return 1  # Safe
    elif (row["TDS"] < 1000) and (row["Turbidity"] < 8):
        return 2  # Needs Purification
    else:
        return 0  # Not Safe

df["Label"] = df.apply(label_quality, axis=1)

# -----------------------
# Step 3: Train/Test Split
# -----------------------
X = df[["pH", "TDS", "Hardness", "Turbidity"]]
y = df["Label"]

if len(df) > 1:  # ensure we have enough rows
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    if len(y_test) > 0:
        print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
else:
    model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X, y)
    print("Model trained on small dataset (no test split).")

# -----------------------
# Step 4: Prediction Function
# -----------------------
def water_quality_status(pH, TDS, Hardness, Turbidity):
    input_data = pd.DataFrame([[pH, TDS, Hardness, Turbidity]],
                              columns=["pH", "TDS", "Hardness", "Turbidity"])
    prediction = model.predict(input_data)[0]
    status_map = {0: "❌ Not Safe", 1: "✅ Safe", 2: "⚠️ Needs Purification"}
    return prediction, status_map[prediction]

# -----------------------
# Step 5: IoT Device Trigger
# -----------------------
def trigger_iot_device(action):
    """Simulate IoT trigger. Replace with MQTT or API call in real life."""
    print(f"[IoT TRIGGER] Purifier Action: {action}")

# -----------------------
# Step 6: Workflow Simulation
# -----------------------
if __name__ == "__main__":
    # IoT sensor readings
    iot_values = {"pH": 6.2, "TDS": 950, "Hardness": 400, "Turbidity": 6.5}
    print("\n--- IoT Device Data ---")
    print("Raw Water Quality:", iot_values)
    pred, status = water_quality_status(**iot_values)
    print("Status:", status)

    # User-modified values
    print("\n--- User Modified Data ---")
    user_modified = {"pH": 7.2, "TDS": 450, "Hardness": 180, "Turbidity": 2.0}
    print("Adjusted Water Quality:", user_modified)
    pred, status = water_quality_status(**user_modified)
    print("Status:", status)

    # Trigger IoT action
    if pred != 1:
        trigger_iot_device("Start Purification")
    else:
        trigger_iot_device("No Action Needed")
