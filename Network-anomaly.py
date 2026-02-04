import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

# =========================
# Load Dataset
# =========================

DATASET_PATH = "KDD+.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'level'
]

df = pd.read_csv(DATASET_PATH, names=columns)

# =========================
# Label Engineering
# =========================

df['attack_flag'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)

dos_attacks = [
    'apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
    'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'
]
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = [
    'buffer_overflow', 'loadmdoule', 'perl', 'ps',
    'rootkit', 'sqlattack', 'xterm'
]
access_attacks = [
    'ftp_write', 'guess_passwd', 'http_tunnel', 'imap',
    'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
    'snmpguess', 'spy', 'warezclient', 'warezmaster',
    'xclock', 'xsnoop'
]

def map_attack(attack):
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in privilege_attacks:
        return 3
    elif attack in access_attacks:
        return 4
    return 0

df['attack_map'] = df['attack'].apply(map_attack)

# =========================
# Feature Engineering
# =========================

categorical_features = ['protocol_type', 'service']
encoded_cat = pd.get_dummies(df[categorical_features])

numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

X = encoded_cat.join(df[numeric_features])
y = df['attack_map']

# =========================
# Train / Validation / Test Split
# =========================

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=1337, stratify=y
)

train_X, val_X, train_y, val_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1337, stratify=train_y
)

# =========================
# Model Training
# =========================

rf_model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=1337
)

rf_model.fit(train_X, train_y)

# =========================
# Evaluation
# =========================

def evaluate(model, X, y, name="Dataset"):
    preds = model.predict(X)
    print(f"\n{name} Evaluation")
    print("Accuracy :", accuracy_score(y, preds))
    print("Precision:", precision_score(y, preds, average="weighted"))
    print("Recall   :", recall_score(y, preds, average="weighted"))
    print("F1-Score :", f1_score(y, preds, average="weighted"))
    print("\nClassification Report:\n")
    print(classification_report(y, preds))

evaluate(rf_model, val_X, val_y, "Validation Set")
evaluate(rf_model, test_X, test_y, "Test Set")

# =========================
# Save Model
# =========================

joblib.dump(rf_model, "network_anomaly_detection_model.joblib")
print("\nModel saved as network_anomaly_detection_model.joblib")
