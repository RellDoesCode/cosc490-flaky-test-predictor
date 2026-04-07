# Flaky Test Prediction Without Re-Runs

**Towson University — COSC 490, Spring 2026**  
Terrell Beasley, Fallon Katz, Gregory Lawton, Oluwaseyi Salisu, Zaynab Tabassi

## Overview

This project builds a machine learning pipeline to predict flaky tests in Java software projects using static code features extracted directly from test source files — no runtime instrumentation or test reruns required.

We train Random Forest and XGBoost classifiers on the [FlakeFlagger dataset](https://github.com/AlshammariA/FlakeFlagger) and evaluate performance using stratified 5-fold cross-validation.

---

## Project Structure

```
cosc490-flaky-test-predictor/
├── src/
│   ├── main.py                     # Run baseline pipeline (FlakeFlagger pre-extracted features)
│   ├── static_feature_extractor.py # Extract static features from raw Java test files
│   ├── dataset_loader.py           # Load and parse the CSV dataset
│   ├── data_cleaning.py            # Remove duplicates, standardize labels
│   ├── feature_extractor.py        # Prepare feature matrix for model training
│   ├── models.py                   # Train Random Forest and XGBoost
│   ├── evaluation.py               # Stratified cross-validation and metrics
│   ├── shap_analysis.py            # SHAP feature importance (optional)
│   └── stats.py                    # Dataset statistics and class balance check
├── data/
│   └── flakeflagger/
│       ├── processed_data.csv      # FlakeFlagger dataset (pre-extracted features + labels)
│       ├── static_features.csv     # Our extracted static features (committed)
│       └── test_files/             # Raw Java test files — NOT committed (see below)
├── experiments/
│   └── cross_project_tests.py      # Cross-project evaluation (in progress)
├── results/                        # Output results
├── cleaned_dataset.xlsx            # Cleaned dataset export
└── dependencies.txt                # Required Python packages
```

---

## Setup

### 1. Install dependencies

```bash
pip install pandas scikit-learn xgboost openpyxl
```

### 2. Run the baseline pipeline

Uses the FlakeFlagger pre-extracted features (no raw Java files needed):

```bash
python -m src.main
```

### 3. Run on static features

The extracted static features are already committed at `data/flakeflagger/static_features.csv`. To train directly on them:

```bash
python -c "
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

df = pd.read_csv('data/flakeflagger/static_features.csv')
X = df.drop(columns=['project', 'test_name', 'label'])
y = df['label']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
print('RF F1:', cross_val_score(rf, X, y, cv=cv, scoring='f1').mean().round(3))
"
```

---

## Getting the Raw Java Test Files (Optional)

The raw `.java` test files are **not committed** to this repo because they total 19,000+ files across 24 open-source projects and would make the repo several gigabytes. The extracted output (`static_features.csv`) is already committed and is all you need to run the models.

If you want to re-run or modify the static feature extractor, follow these steps:

### Step 1 — Enable long path support (Windows only)

Run this in Git Bash before cloning anything. Some Java repos have deeply nested paths that exceed Windows' 260-character limit:

```bash
git config --global core.longpaths true
```

### Step 2 — Create the test_files directory

```bash
mkdir -p data/flakeflagger/test_files
```

### Step 3 — Clone each project using sparse checkout

For each of the 24 projects, clone only the test directories to avoid downloading the full repo. Example for logback:

```bash
git clone --filter=blob:none --sparse https://github.com/qos-ch/logback data/flakeflagger/test_files/qos-ch-logback
cd data/flakeflagger/test_files/qos-ch-logback
git sparse-checkout set src/test/java
git checkout
cd ../../../..
```

### Step 4 — Full project list

The mapping between project names in the CSV and their GitHub repos is defined in `src/static_feature_extractor.py` under `PROJECT_FOLDER_MAP`:

| CSV Project Name | GitHub Repo | Folder Name |
|---|---|---|
| logback | github.com/qos-ch/logback | qos-ch-logback |
| incubator-dubbo | github.com/apache/incubator-dubbo | apache-incubator-dubbo |
| handlebars.java | github.com/jknack/handlebars.java | jknack-handlebars |
| http-request | github.com/kevinsawicki/http-request | kevinsawicki-http-request |
| jimfs | github.com/google/jimfs | google-jimfs |
| zxing | github.com/zxing/zxing | zxing-zxing |
| hector | github.com/hector-client/hector | hector-client-hector |
| okhttp | github.com/square/okhttp | square-okhttp |
| ninja | github.com/ninjaframework/ninja | ninjaframework-ninja |
| achilles | github.com/doanduyhai/Achilles | doanduyhai-Achilles |
| elastic-job-lite | github.com/elasticjob/elastic-job-lite | elasticjob-elastic-job-lite |
| undertow | github.com/undertow-io/undertow | undertow-io-undertow |
| activiti | github.com/Activiti/Activiti | activiti-activiti |
| ambari | github.com/apache/ambari | apache-ambari |
| commons-exec | github.com/apache/commons-exec | apache-commons-exec |
| hbase | github.com/apache/hbase | apache-hbase |
| httpcore | github.com/apache/httpcomponents-core | apache-httpcore |
| assertj-core | github.com/joel-costigliola/assertj-core | joel-costigliola-assertj-core |
| java-websocket | github.com/TooTallNate/java-websocket | tootallnate-java-websocket |
| wildfly | github.com/wildfly/wildfly | wildfly-wildfly |
| spring-boot | github.com/spring-projects/spring-boot | spring-projects-spring-boot |
| wro4j | github.com/wro4j/wro4j | wro4j-wro4j |
| alluxio | github.com/Alluxio/alluxio | Alluxio-alluxio |
| orbit | github.com/orbit/orbit | orbit-orbit |

> **Note:** Multi-module Maven projects (spring-boot, wildfly, activiti, etc.) have test files nested under module subdirectories rather than directly under `src/test/java`. The sparse checkout paths need to be set per module. Use `git ls-tree -r HEAD --name-only | grep src/test/java` inside the cloned repo to find the correct paths, then set sparse checkout accordingly.

### Step 5 — Re-run the extractor

Once the files are in place:

```bash
python -m src.static_feature_extractor
```

This regenerates `data/flakeflagger/static_features.csv`.

---

## Results

### Baseline — FlakeFlagger pre-extracted features (with runtime data)

Evaluated on 22,236 tests (811 flaky, 3.65%) — stratified 5-fold CV:

| Model | Avg F1 | Precision (flaky) | Recall (flaky) |
|---|---|---|---|
| Random Forest | 0.677 | 0.76 | 0.61 |
| XGBoost | 0.682 | 0.83 | 0.58 |

### Our approach — Static features only (no runtime instrumentation)

Evaluated on 12,699 tests (269 flaky, 2.12%) matched across 19 of 24 projects — stratified 5-fold CV:

| Model | Avg F1 | Precision (flaky) | Recall (flaky) |
|---|---|---|---|
| Random Forest | 0.443 | 0.31 | 0.75 |
| XGBoost | 0.659 | 0.84 | 0.54 |

XGBoost on static-only features (F1=0.659) comes within 0.023 of the instrumented baseline (F1=0.682), demonstrating near-equivalent prediction without any runtime data.

---

## Static Features Extracted

| Feature | Description |
|---|---|
| `loc` | Non-blank lines of code |
| `num_asserts` | Number of assert statements |
| `thread_sleep_count` | Number of `Thread.sleep()` calls |
| `has_thread_sleep` | Binary: any `Thread.sleep()` present |
| `async_wait_count` | Count of async/sync primitives (`await`, `CountDownLatch`, etc.) |
| `has_async_wait` | Binary: any async wait present |
| `has_file_io` | Binary: file system access detected |
| `has_network_io` | Binary: network access detected |
| `has_concurrency` | Binary: concurrency APIs detected |
| `num_test_methods` | Number of `@Test` annotated methods |
| `num_try_catch` | Number of catch blocks |
| `has_setup_teardown` | Binary: `@Before`/`@After` hooks present |
| `num_conditionals` | Number of `if`/`for`/`while`/`switch` statements |
| `has_random` | Binary: `Random` or `Math.random()` usage |
| `has_system_time` | Binary: system time access detected |
| `num_annotations` | Total annotation count |
