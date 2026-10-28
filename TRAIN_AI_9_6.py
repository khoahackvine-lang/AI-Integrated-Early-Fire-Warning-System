import sys
import io

# Ensure UTF-8 output encoding for console compatibility
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. ĐỌC DATA
# =====================================================

train_df = pd.read_excel("fire_train_70.xlsx")
val_df = pd.read_excel("fire_validation_10.xlsx")
test_df = pd.read_excel("fire_test_20.xlsx")


# =====================================================
# 2. FEATURE VÀ LABEL
# =====================================================

FEATURES = [
    "temp_c",
    "hum_pct",
    "mq2_raw",
    "mq9_raw",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5"
]

TARGET = "label"

NORMAL_LABEL = 0
SMOKE_LABEL = 1
FIRE_LABEL = 2

CLASS_NAMES = {
    0: "Normal",
    1: "Smoke",
    2: "Fire"
}

AUGMENT_NOISE_FEATURES = [
    "temp_c",
    "hum_pct",
    "mq2_raw",
    "mq9_raw"
]

AUGMENT_ENABLED = True
AUGMENT_EXTRA_RATIO = 0.20
AUGMENT_NOISE_LEVEL = 0.02
RANDOM_STATE = 42


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# =====================================================
# 3. DATA AUGMENTATION CHO TẬP TRAIN
# =====================================================

def augment_tabular_sensor_data(
    X,
    y,
    noise_features,
    extra_ratio=0.20,
    noise_level=0.02,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    X_aug_parts = [X.copy()]
    y_aug_parts = [y.copy()]

    label_counts = y.value_counts().sort_index()

    print("\n================ PHÂN BỐ LABEL TRƯỚC AUGMENTATION ================")
    print(label_counts)

    for label, count in label_counts.items():

        n_new = int(count * extra_ratio)

        if n_new <= 0:
            continue

        X_class = X[y == label].copy()

        sampled_idx = rng.choice(
            X_class.index.to_numpy(),
            size=n_new,
            replace=True
        )

        X_new = X_class.loc[sampled_idx].copy().reset_index(drop=True)

        for col in noise_features:
            if col in X_new.columns:
                std_value = X[col].std()

                if pd.isna(std_value) or std_value == 0:
                    std_value = 1.0

                noise = rng.normal(
                    loc=0.0,
                    scale=std_value * noise_level,
                    size=len(X_new)
                )

                X_new[col] = X_new[col] + noise

        y_new = pd.Series([label] * n_new, name=y.name)

        X_aug_parts.append(X_new)
        y_aug_parts.append(y_new)

    X_aug = pd.concat(X_aug_parts, axis=0).reset_index(drop=True)
    y_aug = pd.concat(y_aug_parts, axis=0).reset_index(drop=True)

    print("\n================ PHÂN BỐ LABEL SAU AUGMENTATION ================")
    print(y_aug.value_counts().sort_index())

    print("\nSố mẫu train ban đầu:", len(X))
    print("Số mẫu train sau augmentation:", len(X_aug))

    return X_aug, y_aug


if AUGMENT_ENABLED:
    X_train_model, y_train_model = augment_tabular_sensor_data(
        X_train,
        y_train,
        noise_features=AUGMENT_NOISE_FEATURES,
        extra_ratio=AUGMENT_EXTRA_RATIO,
        noise_level=AUGMENT_NOISE_LEVEL,
        random_state=RANDOM_STATE
    )
else:
    X_train_model = X_train.copy()
    y_train_model = y_train.copy()


# =====================================================
# 4. MODEL SO SÁNH
# =====================================================

models = {
    "MLP (16,8)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42
        ))
    ]),

    "Gaussian NB": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "Dummy Classifier": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DummyClassifier(
            strategy="most_frequent",
            random_state=42
        ))
    ])
}


# =====================================================
# 5. HÀM ĐÁNH GIÁ
# =====================================================

def one_class_metric(metric_func, y_true, y_pred, label, **kwargs):
    return metric_func(
        y_true,
        y_pred,
        labels=[label],
        average="macro",
        zero_division=0,
        **kwargs
    )


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),

        "precision_macro": precision_score(
            y, y_pred, average="macro", zero_division=0
        ),

        "recall_macro": recall_score(
            y, y_pred, average="macro", zero_division=0
        ),

        "f1_macro": f1_score(
            y, y_pred, average="macro", zero_division=0
        ),

        "f2_macro": fbeta_score(
            y, y_pred, beta=2, average="macro", zero_division=0
        ),

        "normal_precision": one_class_metric(
            precision_score, y, y_pred, NORMAL_LABEL
        ),
        "normal_recall": one_class_metric(
            recall_score, y, y_pred, NORMAL_LABEL
        ),
        "normal_f1": one_class_metric(
            f1_score, y, y_pred, NORMAL_LABEL
        ),
        "normal_f2": one_class_metric(
            fbeta_score, y, y_pred, NORMAL_LABEL, beta=2
        ),

        "smoke_precision": one_class_metric(
            precision_score, y, y_pred, SMOKE_LABEL
        ),
        "smoke_recall": one_class_metric(
            recall_score, y, y_pred, SMOKE_LABEL
        ),
        "smoke_f1": one_class_metric(
            f1_score, y, y_pred, SMOKE_LABEL
        ),
        "smoke_f2": one_class_metric(
            fbeta_score, y, y_pred, SMOKE_LABEL, beta=2
        ),

        "fire_precision": one_class_metric(
            precision_score, y, y_pred, FIRE_LABEL
        ),
        "fire_recall": one_class_metric(
            recall_score, y, y_pred, FIRE_LABEL
        ),
        "fire_f1": one_class_metric(
            f1_score, y, y_pred, FIRE_LABEL
        ),
        "fire_f2": one_class_metric(
            fbeta_score, y, y_pred, FIRE_LABEL, beta=2
        ),

        "y_pred": y_pred
    }


def check_overfitting(train_f1, val_f1, test_f1):
    gap_train_val = train_f1 - val_f1
    gap_train_test = train_f1 - test_f1

    if gap_train_val > 0.10 or gap_train_test > 0.10:
        status = "Overfitting nặng"
    elif gap_train_val > 0.05 or gap_train_test > 0.05:
        status = "Có dấu hiệu overfitting nhẹ"
    else:
        status = "Không overfit rõ ràng"

    return status, gap_train_val, gap_train_test


def print_metric_block(name, train_result, val_result, test_result, overfit_status):
    print("\n" + "=" * 90)
    print(f"MODEL: {name}")
    print("=" * 90)

    print(
        f"{'Dataset':<12}"
        f"{'Acc':>8}"
        f"{'Pre':>8}"
        f"{'Rec':>8}"
        f"{'F1':>8}"
        f"{'F2':>8}"
        f"{'Nor_Rec':>10}"
        f"{'Smo_Rec':>10}"
        f"{'Fire_Rec':>10}"
    )

    print("-" * 90)

    for dataset_name, result in [
        ("Train", train_result),
        ("Validation", val_result),
        ("Test", test_result)
    ]:
        print(
            f"{dataset_name:<12}"
            f"{result['accuracy']:>8.4f}"
            f"{result['precision_macro']:>8.4f}"
            f"{result['recall_macro']:>8.4f}"
            f"{result['f1_macro']:>8.4f}"
            f"{result['f2_macro']:>8.4f}"
            f"{result['normal_recall']:>10.4f}"
            f"{result['smoke_recall']:>10.4f}"
            f"{result['fire_recall']:>10.4f}"
        )

    print("-" * 90)
    print(f"Overfit status: {overfit_status}")


# =====================================================
# 6. TRAIN VÀ ĐÁNH GIÁ
# =====================================================

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train_model, y_train_model)

    train_result = evaluate_model(model, X_train, y_train)
    val_result = evaluate_model(model, X_val, y_val)
    test_result = evaluate_model(model, X_test, y_test)

    overfit_status, gap_train_val, gap_train_test = check_overfitting(
        train_result["f1_macro"],
        val_result["f1_macro"],
        test_result["f1_macro"]
    )

    print_metric_block(
        name,
        train_result,
        val_result,
        test_result,
        overfit_status
    )

    results.append({
        "model": name,

        "train_accuracy": train_result["accuracy"],
        "train_precision": train_result["precision_macro"],
        "train_recall": train_result["recall_macro"],
        "train_f1": train_result["f1_macro"],
        "train_f2": train_result["f2_macro"],

        "val_accuracy": val_result["accuracy"],
        "val_precision": val_result["precision_macro"],
        "val_recall": val_result["recall_macro"],
        "val_f1": val_result["f1_macro"],
        "val_f2": val_result["f2_macro"],

        "test_accuracy": test_result["accuracy"],
        "test_precision": test_result["precision_macro"],
        "test_recall": test_result["recall_macro"],
        "test_f1": test_result["f1_macro"],
        "test_f2": test_result["f2_macro"],

        "test_normal_precision": test_result["normal_precision"],
        "test_normal_recall": test_result["normal_recall"],
        "test_normal_f1": test_result["normal_f1"],
        "test_normal_f2": test_result["normal_f2"],

        "test_smoke_precision": test_result["smoke_precision"],
        "test_smoke_recall": test_result["smoke_recall"],
        "test_smoke_f1": test_result["smoke_f1"],
        "test_smoke_f2": test_result["smoke_f2"],

        "test_fire_precision": test_result["fire_precision"],
        "test_fire_recall": test_result["fire_recall"],
        "test_fire_f1": test_result["fire_f1"],
        "test_fire_f2": test_result["fire_f2"],

        "gap_train_val_f1": gap_train_val,
        "gap_train_test_f1": gap_train_test,
        "overfit_status": overfit_status
    })

    trained_models[name] = model


# =====================================================
# 7. BẢNG SO SÁNH MODEL
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "test_fire_recall",
        "test_smoke_recall",
        "test_f2",
        "test_f1",
        "test_accuracy"
    ],
    ascending=False
).reset_index(drop=True)

summary_cols = [
    "model",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_f2",
    "test_normal_recall",
    "test_smoke_recall",
    "test_fire_recall",
    "gap_train_test_f1",
    "overfit_status"
]

summary_df = results_df[summary_cols].copy()

for col in summary_df.columns:
    if col not in ["model", "overfit_status"]:
        summary_df[col] = summary_df[col].round(4)

print("\n\n" + "=" * 120)
print("BẢNG SO SÁNH MODEL")
print("=" * 120)
print(summary_df.to_string(index=False))


# =====================================================
# 8. BƯỚC 7 - TÀI LIỆU MÔ TẢ DATASET
# =====================================================

label_distribution_df = pd.DataFrame({
    "train_original": y_train.value_counts().sort_index(),
    "train_after_augmentation": y_train_model.value_counts().sort_index(),
    "validation": y_val.value_counts().sort_index(),
    "test": y_test.value_counts().sort_index()
}).fillna(0).astype(int)

dataset_overview_df = pd.DataFrame([
    {
        "item": "Dataset name",
        "description": "Fire Detection Sensor Dataset"
    },
    {
        "item": "Problem type",
        "description": "Supervised Learning - Multi-class Classification"
    },
    {
        "item": "Input data type",
        "description": "Tabular sensor data collected from ESP32-based IoT system"
    },
    {
        "item": "Number of classes",
        "description": "3 classes: Normal, Smoke, Fire"
    },
    {
        "item": "Train file",
        "description": "fire_train_70.xlsx"
    },
    {
        "item": "Validation file",
        "description": "fire_validation_10.xlsx"
    },
    {
        "item": "Test file",
        "description": "fire_test_20.xlsx"
    },
    {
        "item": "Train/Validation/Test ratio",
        "description": "70% / 10% / 20%"
    }
])

feature_description_df = pd.DataFrame([
    {"feature": "temp_c", "description": "Temperature value measured by DHT22 sensor, unit: Celsius"},
    {"feature": "hum_pct", "description": "Humidity value measured by DHT22 sensor, unit: percent"},
    {"feature": "mq2_raw", "description": "Raw analog value from MQ2 gas/smoke sensor"},
    {"feature": "mq9_raw", "description": "Raw analog value from MQ9 gas sensor"},
    {"feature": "f1", "description": "Digital/processed value from flame sensor 1"},
    {"feature": "f2", "description": "Digital/processed value from flame sensor 2"},
    {"feature": "f3", "description": "Digital/processed value from flame sensor 3"},
    {"feature": "f4", "description": "Digital/processed value from flame sensor 4"},
    {"feature": "f5", "description": "Digital/processed value from flame sensor 5"},
    {"feature": "label", "description": "Target label: 0 = Normal, 1 = Smoke, 2 = Fire"}
])

label_description_df = pd.DataFrame([
    {"label": 0, "class_name": "Normal", "description": "Normal condition without smoke or fire"},
    {"label": 1, "class_name": "Smoke", "description": "Smoke condition detected by sensor values"},
    {"label": 2, "class_name": "Fire", "description": "Fire condition detected by flame/gas/temperature sensor values"}
])

preprocessing_df = pd.DataFrame([
    {
        "step": "Feature selection",
        "description": "Select 9 input features: temp_c, hum_pct, mq2_raw, mq9_raw, f1, f2, f3, f4, f5"
    },
    {
        "step": "Data augmentation",
        "description": "Apply small Gaussian noise to analog sensor features in the training set only"
    },
    {
        "step": "Standardization",
        "description": "Use StandardScaler in Pipeline to standardize input features before model training"
    },
    {
        "step": "No leakage rule",
        "description": "Validation and test sets are not augmented and are only used for evaluation"
    }
])

usage_guideline_df = pd.DataFrame([
    {
        "item": "How to train",
        "description": "Run this Python script in the same folder as the three Excel dataset files"
    },
    {
        "item": "How to evaluate",
        "description": "Use Accuracy, Precision, Recall, F1-score, F2-score, class recall and confusion matrix"
    },
    {
        "item": "How to deploy",
        "description": "Use the saved mlp_fire_detection_model.pkl or extract MLP weights for ESP32 deployment"
    },
    {
        "item": "Limitation",
        "description": "The dataset is collected from specific sensors and scenarios, so performance may vary in new environments"
    }
])


def create_dataset_documentation_text():
    text = f"""
DATASET DOCUMENTATION - FIRE DETECTION SENSOR DATASET
====================================================

1. Dataset Overview
-------------------
Dataset name: Fire Detection Sensor Dataset
Problem type: Supervised Learning - Multi-class Classification
Input data type: Tabular sensor data
Application: Fire detection using ESP32 and sensor system

The dataset is used to train an AI model that classifies sensor data into three states:
0 = Normal
1 = Smoke
2 = Fire

2. Data Source and Collection
-----------------------------
The data is collected from an ESP32-based IoT system using multiple sensors:
- DHT22 for temperature and humidity
- MQ2 for smoke/gas signal
- MQ9 for gas signal
- Flame sensors f1, f2, f3, f4, f5

The collected data is stored in Excel files and divided into:
- fire_train_70.xlsx
- fire_validation_10.xlsx
- fire_test_20.xlsx

3. Feature Description
----------------------
Input features:
- temp_c: temperature value in Celsius
- hum_pct: humidity value in percent
- mq2_raw: raw analog value from MQ2 sensor
- mq9_raw: raw analog value from MQ9 sensor
- f1, f2, f3, f4, f5: flame sensor values

Target label:
- label: output class of the sample

4. Label Description
--------------------
0 = Normal: normal environment without smoke or fire
1 = Smoke : smoke condition
2 = Fire  : fire condition

5. Preprocessing Pipeline
-------------------------
The preprocessing process includes:
- Feature selection: select 9 sensor features
- Data augmentation: add small Gaussian noise to analog sensor values in training data only
- Standardization: use StandardScaler to normalize feature scale before training

The validation and test datasets are not augmented. They are kept unchanged to ensure objective evaluation.

6. Data Augmentation
--------------------
Data augmentation is applied only to the training set.
Small Gaussian noise is added to analog features:
- temp_c
- hum_pct
- mq2_raw
- mq9_raw

Digital flame sensor features f1-f5 are kept unchanged.
The labels are also kept unchanged.

Augmentation enabled: {AUGMENT_ENABLED}
Extra ratio: {AUGMENT_EXTRA_RATIO}
Noise level: {AUGMENT_NOISE_LEVEL}

7. Data Split
-------------
The dataset is divided into:
- Train: 70%
- Validation: 10%
- Test: 20%

Train original samples: {len(X_train)}
Train after augmentation: {len(X_train_model)}
Validation samples: {len(X_val)}
Test samples: {len(X_test)}

8. Model Usage
--------------
The dataset is used to train and compare:
- MLP (16,8)
- Gaussian Naive Bayes
- Dummy Classifier

The selected model is MLP (16,8).

9. Evaluation Metrics
---------------------
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- F2-score
- Normal Recall
- Smoke Recall
- Fire Recall
- Confusion Matrix

10. Limitations
---------------
The dataset is collected from a specific sensor system and specific experimental conditions.
When deployed in a different environment, the model may require additional data collection and retraining.

11. Output Files
----------------
This training script generates:
- mlp_fire_detection_model.pkl
- model_comparison_results.xlsx
- mlp_prediction_probabilities.xlsx
- dataset_documentation.txt
"""
    return text.strip()


dataset_documentation_text = create_dataset_documentation_text()

with open("dataset_documentation.txt", "w", encoding="utf-8") as f:
    f.write(dataset_documentation_text)

dataset_documentation_df = pd.DataFrame(
    {"dataset_documentation": dataset_documentation_text.split("\n")}
)


# =====================================================
# 9. LƯU KẾT QUẢ RA EXCEL
# =====================================================

with pd.ExcelWriter("model_comparison_results.xlsx", engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="full_results", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    label_distribution_df.to_excel(writer, sheet_name="label_distribution")
    dataset_overview_df.to_excel(writer, sheet_name="dataset_overview", index=False)
    feature_description_df.to_excel(writer, sheet_name="feature_description", index=False)
    label_description_df.to_excel(writer, sheet_name="label_description", index=False)
    preprocessing_df.to_excel(writer, sheet_name="preprocessing", index=False)
    usage_guideline_df.to_excel(writer, sheet_name="usage_guideline", index=False)
    dataset_documentation_df.to_excel(writer, sheet_name="dataset_documentation", index=False)


# =====================================================
# 10. CHỌN MODEL CHÍNH LÀ MLP
# =====================================================

main_model_name = "MLP (16,8)"
main_model = trained_models[main_model_name]
mlp_row = results_df[results_df["model"] == main_model_name].iloc[0]

y_pred_test = main_model.predict(X_test)

print("\n\n" + "=" * 80)
print("MODEL ĐƯỢC CHỌN")
print("=" * 80)

print(f"Model chính        : {main_model_name}")
print(f"Hidden layer       : {main_model.named_steps['model'].hidden_layer_sizes}")
print(f"Test Accuracy      : {mlp_row['test_accuracy']:.4f}")
print(f"Test F1-score      : {mlp_row['test_f1']:.4f}")
print(f"Test F2-score      : {mlp_row['test_f2']:.4f}")
print(f"Normal Recall      : {mlp_row['test_normal_recall']:.4f}")
print(f"Smoke Recall       : {mlp_row['test_smoke_recall']:.4f}")
print(f"Fire Recall        : {mlp_row['test_fire_recall']:.4f}")
print(f"Overfit            : {mlp_row['overfit_status']}")

print("\nClassification Report trên tập Test của MLP:")
print(classification_report(y_test, y_pred_test, zero_division=0))

print("\nConfusion Matrix trên tập Test của MLP:")
print(confusion_matrix(y_test, y_pred_test))


# =====================================================
# 11. LƯU XÁC SUẤT DỰ ĐOÁN MLP
# =====================================================

proba_test = main_model.predict_proba(X_test)
classes = main_model.named_steps["model"].classes_

proba_df = pd.DataFrame(
    proba_test,
    columns=[f"prob_{CLASS_NAMES.get(c, c)}" for c in classes]
)

proba_df["true_label"] = y_test.values
proba_df["pred_label"] = y_pred_test
proba_df["true_name"] = proba_df["true_label"].map(CLASS_NAMES)
proba_df["pred_name"] = proba_df["pred_label"].map(CLASS_NAMES)
proba_df["correct"] = proba_df["true_label"] == proba_df["pred_label"]

proba_df.to_excel("mlp_prediction_probabilities.xlsx", index=False)


# =====================================================
# 12. LƯU MODEL MLP
# =====================================================

model_info = {
    "model_name": main_model_name,
    "hidden_layer_sizes": (16, 8),
    "model": main_model,
    "features": FEATURES,
    "target": TARGET,
    "class_names": CLASS_NAMES,
    "augmentation_enabled": AUGMENT_ENABLED,
    "augmentation_noise_features": AUGMENT_NOISE_FEATURES,
    "augmentation_extra_ratio": AUGMENT_EXTRA_RATIO,
    "augmentation_noise_level": AUGMENT_NOISE_LEVEL,
    "dataset_documentation": dataset_documentation_text,
    "results": results_df
}

joblib.dump(model_info, "mlp_fire_detection_model.pkl")


# =====================================================
# 13. KẾT LUẬN
# =====================================================

print("\n\n" + "=" * 80)
print("KẾT LUẬN")
print("=" * 80)

print(
    f"Mô hình MLP (16,8) được chọn làm mô hình chính. "
    f"Dữ liệu train được tăng cường bằng cách thêm nhiễu nhỏ vào các cảm biến analog "
    f"nhưng vẫn giữ nguyên nhãn ban đầu. "
    f"Trên tập test, mô hình đạt Accuracy = {mlp_row['test_accuracy']:.4f}, "
    f"Precision = {mlp_row['test_precision']:.4f}, "
    f"Recall = {mlp_row['test_recall']:.4f}, "
    f"F1-score = {mlp_row['test_f1']:.4f}, "
    f"F2-score = {mlp_row['test_f2']:.4f}, "
    f"Normal Recall = {mlp_row['test_normal_recall']:.4f}, "
    f"Smoke Recall = {mlp_row['test_smoke_recall']:.4f}, "
    f"Fire Recall = {mlp_row['test_fire_recall']:.4f}. "
    f"Kết quả cho thấy mô hình {mlp_row['overfit_status'].lower()}."
)

print("\nĐã lưu:")
print("- mlp_fire_detection_model.pkl")
print("- model_comparison_results.xlsx")
print("- mlp_prediction_probabilities.xlsx")
print("- dataset_documentation.txt")