import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# CONFIGURATION
# ==================================================

DATA_PATH = "dataset/training_data.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==================================================
# HEADER
# ==================================================

print("\n" + "=" * 65)
print("       AI RESUME ANALYZER - PROPER MODEL EVALUATION")
print("=" * 65)


# ==================================================
# 1. LOAD DATASET
# ==================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Columns: {list(df.columns)}"
)


# ==================================================
# 2. CHECK REQUIRED COLUMNS
# ==================================================

text_column = None
label_column = None


possible_text_columns = [
    "Resume Text",
    "Resume",
    "resume_text",
    "resume",
    "Text",
    "text"
]

possible_label_columns = [
    "Job Role",
    "Job Title",
    "job_role",
    "job title",
    "Label",
    "label"
]


for column in possible_text_columns:

    if column in df.columns:

        text_column = column
        break


for column in possible_label_columns:

    if column in df.columns:

        label_column = column
        break


if text_column is None:

    raise ValueError(
        "Resume text column not found.\n"
        f"Available columns: {list(df.columns)}"
    )


if label_column is None:

    raise ValueError(
        "Job-role column not found.\n"
        f"Available columns: {list(df.columns)}"
    )


print(
    f"\nText column : {text_column}"
)

print(
    f"Label column: {label_column}"
)


# ==================================================
# 3. CLEAN DATA
# ==================================================

df = df[
    [
        text_column,
        label_column
    ]
].dropna()

df[text_column] = (
    df[text_column]
    .astype(str)
    .str.strip()
)

df[label_column] = (
    df[label_column]
    .astype(str)
    .str.strip()
)


# Remove empty rows

df = df[
    (df[text_column] != "")
    &
    (df[label_column] != "")
].reset_index(
    drop=True
)


print(
    f"\nUsable samples: {len(df)}"
)

print(
    f"Unique job roles: "
    f"{df[label_column].nunique()}"
)


# ==================================================
# 4. DISPLAY CLASS DISTRIBUTION
# ==================================================

print("\nJob-role distribution:")

print(
    df[label_column]
    .value_counts()
    .to_string()
)


# ==================================================
# 5. TRAIN / TEST SPLIT
# ==================================================

print("\n" + "=" * 65)
print("[2] Creating train/test split...")
print("=" * 65)

X = df[text_column]
y = df[label_column]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)

print(
    f"Test percentage  : {TEST_SIZE * 100:.0f}%"
)

print(
    f"Random state     : {RANDOM_STATE}"
)


# ==================================================
# 6. TF-IDF
# ==================================================

print("\n" + "=" * 65)
print("[3] Training TF-IDF vectorizer...")
print("=" * 65)


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)


# IMPORTANT:
# Fit ONLY on training data.

X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Transform test data using the
# already-fitted vectorizer.

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    f"\nTraining TF-IDF shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Testing TF-IDF shape : "
    f"{X_test_tfidf.shape}"
)

print(
    f"Vocabulary size      : "
    f"{len(vectorizer.vocabulary_)}"
)


# ==================================================
# 7. TRAIN LINEAR SVM
# ==================================================

print("\n" + "=" * 65)
print("[4] Training Linear SVM...")
print("=" * 65)


base_svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE
)


# ==================================================
# 8. CALIBRATED SVM
# ==================================================

print("\n[5] Calibrating SVM probabilities...")

model = CalibratedClassifierCV(
    estimator=base_svm,
    method="sigmoid",
    cv=5
)


model.fit(
    X_train_tfidf,
    y_train
)


print(
    "✓ Calibrated Linear SVM trained"
)


# ==================================================
# 9. PREDICT TEST SET
# ==================================================

print("\n" + "=" * 65)
print("[6] Evaluating on UNSEEN test data...")
print("=" * 65)


y_pred = model.predict(
    X_test_tfidf
)


# ==================================================
# 10. METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ==================================================
# 11. DISPLAY MAIN RESULTS
# ==================================================

print("\n" + "=" * 65)
print("                 FINAL MODEL RESULTS")
print("=" * 65)

print(
    f"\nAccuracy:           "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Weighted Recall:    "
    f"{recall * 100:.2f}%"
)

print(
    f"Weighted F1 Score:  "
    f"{weighted_f1 * 100:.2f}%"
)


# ==================================================
# 12. CLASSIFICATION REPORT
# ==================================================

print("\n" + "=" * 65)
print("                 CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==================================================
# 13. CONFUSION MATRIX
# ==================================================

labels = sorted(
    set(y_test) | set(y_pred)
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("\n" + "=" * 65)
print("                    CONFUSION MATRIX")
print("=" * 65)

print(
    "\nRole index mapping:"
)

for index, label in enumerate(labels):

    print(
        f"{index:2d} -> {label}"
    )


print("\nMatrix:")

print(cm)


# ==================================================
# 14. ERROR ANALYSIS
# ==================================================

results_df = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": y_pred

})


errors = results_df[
    results_df["Actual"]
    !=
    results_df["Predicted"]
]


print("\n" + "=" * 65)
print("                    ERROR ANALYSIS")
print("=" * 65)

print(
    f"\nIncorrect predictions: "
    f"{len(errors)}"
)

print(
    f"Correct predictions:   "
    f"{len(results_df) - len(errors)}"
)


if len(errors) > 0:

    print(
        "\nFirst 20 incorrect predictions:"
    )

    print(
        errors
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "\n✓ No incorrect predictions "
        "in the test set."
    )


# ==================================================
# 15. SAVE EVALUATION RESULTS
# ==================================================

evaluation_results = {

    "dataset_size":
        len(df),

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test),

    "number_of_roles":
        df[label_column].nunique(),

    "accuracy":
        round(
            accuracy * 100,
            4
        ),

    "weighted_precision":
        round(
            precision * 100,
            4
        ),

    "weighted_recall":
        round(
            recall * 100,
            4
        ),

    "weighted_f1":
        round(
            weighted_f1 * 100,
            4
        )
}


results_path = (
    "models/evaluation_results.csv"
)


pd.DataFrame(
    [evaluation_results]
).to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 65)
print("                  EVALUATION COMPLETE")
print("=" * 65)

print(
    f"\n✓ Results saved to:"
    f"\n  {results_path}"
)

print(
    "\nIMPORTANT:"
)

print(
    "These metrics are calculated on the "
    "20% test set that was not used to fit "
    "the TF-IDF vectorizer or SVM model."
)

print(
    "\nDo NOT use the previous 99.83% result "
    "as the final test accuracy."
)

print(
    "Use the results above after reviewing "
    "the train/test methodology."
)