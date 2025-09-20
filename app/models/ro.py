import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import json
import time

# -----------------------
# Load dataset (replace this with IoT logs in production)
# -----------------------
data = {
    "pH": [3.716080075, 8.099124189, 8.316765884],
    "TDS": [18630.05786, 19909.54173, 22018.41744],
    "Hardness": [129.4229205, 224.2362594, 214.3733941],
    "Turbidity": [4.500656275, 3.05593375, 4.628770537]
}
df = pd.DataFrame(data)

# Simulate user-desired TDS values (RO users typically want < 500 ppm)
df["desired_tds"] = np.random.choice([50, 100, 200, 300, 500], size=len(df))

# Compute actuator_percent as a rough ratio of reduction needed
df["actuator_percent"] = np.clip((df["TDS"] - df["desired_tds"]) / (df["TDS"] + 1e-6) * 100, 0, 100)

# Estimate final TDS after applying actuator
df["final_tds"] = df["TDS"] * (1 - df["actuator_percent"]/100) + df["desired_tds"]*0.05

# Create Safe/Not Safe label
def classify(row):
    if row["final_tds"] <= 500 and row["Turbidity"] < 5 and 6.5 <= row["pH"] <= 8.5:
        return 1  # Safe
    else:
        return 0  # Not Safe

df["safe"] = df.apply(classify, axis=1)

print("Dataset:\n", df.head())

# -----------------------
# Train Models
# -----------------------
features = ["TDS", "pH", "Hardness", "Turbidity", "desired_tds"]

X = df[features].values
y_reg = df["actuator_percent"].values
y_clf = df["safe"].values

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = \
    train_test_split(X, y_reg, y_clf, test_size=0.3, random_state=42)

# Regressor for actuator control
rfr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rfr.fit(X_train, y_reg_train)
y_reg_pred = rfr.predict(X_test)
print("Regressor MSE:", mean_squared_error(y_reg_test, y_reg_pred))
print("Regressor R2:", r2_score(y_reg_test, y_reg_pred))

# Classifier for safety
xgb = XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb.fit(X_train, y_clf_train)
y_clf_pred = xgb.predict(X_test)
print("Classifier Accuracy:", accuracy_score(y_clf_test, y_clf_pred))
print("Classification Report:\n", classification_report(y_clf_test, y_clf_pred))

# Save models
joblib.dump(rfr, "rfr_ro.joblib")
joblib.dump(xgb, "xgb_ro_safe.joblib")
print("Models saved: rfr_ro.joblib, xgb_ro_safe.joblib")

# -----------------------
# Prediction Function
# -----------------------
def predict_and_act(models, sensor_payload, desired_tds):
    rfr, xgb = models
    input_data = pd.DataFrame([[sensor_payload["TDS"], sensor_payload["pH"],
                                sensor_payload["Hardness"], sensor_payload["Turbidity"],
                                desired_tds]],
                                columns=["TDS", "pH", "Hardness", "Turbidity", "desired_tds"])

    actuator_percent = float(np.clip(rfr.predict(input_data)[0], 0, 100))
    final_tds = sensor_payload["TDS"] * (1 - actuator_percent/100) + desired_tds*0.05
    safe_flag = int(xgb.predict(input_data)[0])

    action = {
        "actuator_percent": round(actuator_percent, 2),
        "predicted_final_tds": round(final_tds, 2),
        "safe": bool(safe_flag),
        "timestamp": time.time()
    }

    payload = json.dumps({
        "command": "set_actuator",
        "actuator_percent": action["actuator_percent"],
        "desired_tds": desired_tds,
        "predicted_final_tds": action["predicted_final_tds"],
        "safe": action["safe"]
    })
    action["mqtt_payload_preview"] = payload

    return action

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    models = (rfr, xgb)

    # Example IoT reading
    sensor = {
        "TDS": 1800,
        "pH": 7.2,
        "Hardness": 250,
        "Turbidity": 3.5
    }
    desired = 200

    action = predict_and_act(models, sensor, desired)
    print("Action for IoT device:\n", action)
