from pathlib import Path
import pandas as pd
import joblib

LABEL_MAP = {0: "DDoS", 1: "Intrusion", 2: "Malware"}

CATEGORICAL_COLS = [
    "Protocol",
    "Packet Type",
    "Traffic Type",
    "Browser",
    "OS",
    "Device",
    "Engine",
    "Log Source",
    "Categorical Source Port",
    "Categorical Destination Port",
    "asn_source type",
    "asn_dest type",
]

class Predictor:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)
        self.expected_cols = self.model.get_booster().feature_names

    def predict_from_dataframe(self, df: pd.DataFrame) -> str:
        if "Attack Type" in df.columns:
            df = df.drop(columns=["Attack Type"])

        required_cols = set([
        "Source Port", "Destination Port", "Protocol", "Packet Length", "Packet Type",
        "Traffic Type", "Malware Indicators", "Anomaly Scores", "Alerts/Warnings",
        "Attack Signature", "Action Taken", "Severity Level", "Network Segment",
        "Proxy Information", "Firewall Logs", "IDS/IPS Alerts", "Log Source",
        "Year", "Month"])

        missing = sorted(list(required_cols - set(df.columns)))
        if missing:
            raise ValueError(
                "Invalid CSV format. Missing required columns: " + ", ".join(missing)
            )

        df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

        X = df_enc.reindex(columns=self.expected_cols, fill_value=0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        pred = int(self.model.predict(X)[0])
        return LABEL_MAP.get(pred, str(pred))