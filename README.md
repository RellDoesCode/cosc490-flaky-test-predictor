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

The extracted static features are already committed at `data/flakeflagger/static_features.csv`. Switch `main.py` to use it by setting `file_path = "data/flakeflagger/static_features.csv"` and `drop_runtime=False`, then run:

```bash
python -m src.main
```

---

## Getting the Raw Java Test Files (Optional)

The raw `.java` test files are **not committed** to this repo because they total 19,000+ files across 24 open-source projects and would make the repo several gigabytes. The extracted output (`static_features.csv`) is already committed and is all you need to run the models.

If you want to re-run or modify the static feature extractor, follow these steps:

### Step 1 — Enable long path support (Windows only, skip on Mac/Linux)

Some Java repos have deeply nested paths that exceed Windows' 260-character limit. Run this in Git Bash before cloning:

```bash
git config --global core.longpaths true
```

Mac and Linux users can skip this step.

### Step 2 — Create the test_files directory

From the project root, run:

```bash
mkdir -p data/flakeflagger/test_files
```

### Step 3 — Clone all 24 projects

We use sparse checkout to download only the test source files, not the entire repo. Run each block below from the **project root**. Each block clones one project, switches into it, sets the sparse checkout path, pulls the files, then returns to the project root.

```bash
git clone --filter=blob:none --sparse https://github.com/qos-ch/logback data/flakeflagger/test_files/qos-ch-logback
cd data/flakeflagger/test_files/qos-ch-logback && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/apache/incubator-dubbo data/flakeflagger/test_files/apache-incubator-dubbo
cd data/flakeflagger/test_files/apache-incubator-dubbo && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/jknack/handlebars.java data/flakeflagger/test_files/jknack-handlebars
cd data/flakeflagger/test_files/jknack-handlebars && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/kevinsawicki/http-request data/flakeflagger/test_files/kevinsawicki-http-request
cd data/flakeflagger/test_files/kevinsawicki-http-request && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/google/jimfs data/flakeflagger/test_files/google-jimfs
cd data/flakeflagger/test_files/google-jimfs && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/zxing/zxing data/flakeflagger/test_files/zxing-zxing
cd data/flakeflagger/test_files/zxing-zxing && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/hector-client/hector data/flakeflagger/test_files/hector-client-hector
cd data/flakeflagger/test_files/hector-client-hector && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/square/okhttp data/flakeflagger/test_files/square-okhttp
cd data/flakeflagger/test_files/square-okhttp && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/ninjaframework/ninja data/flakeflagger/test_files/ninjaframework-ninja
cd data/flakeflagger/test_files/ninjaframework-ninja && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/doanduyhai/Achilles data/flakeflagger/test_files/doanduyhai-Achilles
cd data/flakeflagger/test_files/doanduyhai-Achilles && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/elasticjob/elastic-job-lite data/flakeflagger/test_files/elasticjob-elastic-job-lite
cd data/flakeflagger/test_files/elasticjob-elastic-job-lite && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/undertow-io/undertow data/flakeflagger/test_files/undertow-io-undertow
cd data/flakeflagger/test_files/undertow-io-undertow && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/Activiti/Activiti data/flakeflagger/test_files/activiti-activiti
cd data/flakeflagger/test_files/activiti-activiti && git sparse-checkout disable && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/apache/ambari data/flakeflagger/test_files/apache-ambari
cd data/flakeflagger/test_files/apache-ambari && git sparse-checkout set ambari-agent/src/test/java ambari-server/src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/apache/commons-exec data/flakeflagger/test_files/apache-commons-exec
cd data/flakeflagger/test_files/apache-commons-exec && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/apache/hbase data/flakeflagger/test_files/apache-hbase
cd data/flakeflagger/test_files/apache-hbase && git sparse-checkout set hbase-client/src/test/java hbase-server/src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/apache/httpcomponents-core data/flakeflagger/test_files/apache-httpcore
cd data/flakeflagger/test_files/apache-httpcore && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/joel-costigliola/assertj-core data/flakeflagger/test_files/joel-costigliola-assertj-core
cd data/flakeflagger/test_files/joel-costigliola-assertj-core && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/TooTallNate/java-websocket data/flakeflagger/test_files/tootallnate-java-websocket
cd data/flakeflagger/test_files/tootallnate-java-websocket && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/wildfly/wildfly data/flakeflagger/test_files/wildfly-wildfly
cd data/flakeflagger/test_files/wildfly-wildfly && git sparse-checkout disable && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/spring-projects/spring-boot data/flakeflagger/test_files/spring-projects-spring-boot
cd data/flakeflagger/test_files/spring-projects-spring-boot && git sparse-checkout disable && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/wro4j/wro4j data/flakeflagger/test_files/wro4j-wro4j
cd data/flakeflagger/test_files/wro4j-wro4j && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/Alluxio/alluxio data/flakeflagger/test_files/Alluxio-alluxio
cd data/flakeflagger/test_files/Alluxio-alluxio && git sparse-checkout set src/test/java && git checkout && cd ../../../..

git clone --filter=blob:none --sparse https://github.com/orbit/orbit data/flakeflagger/test_files/orbit-orbit
cd data/flakeflagger/test_files/orbit-orbit && git sparse-checkout set src/test/java && git checkout && cd ../../../..
```

> **Note on multi-module projects:**
> - **ambari** — test files are in `ambari-agent/` and `ambari-server/` modules (handled above)
> - **hbase** — test files are in `hbase-client/` and `hbase-server/` modules (handled above)
> - **wildfly**, **spring-boot**, **activiti** — have too many modules to enumerate; `git sparse-checkout disable` pulls the full repo for these three. They are large (wildfly ~1GB, spring-boot ~500MB, activiti ~300MB). If disk space is a concern, you can skip them — the extractor will report 0 matches for those projects and continue without errors.

### Step 4 — Re-run the extractor

Once all projects are cloned, from the project root run:

```bash
# Mac / Linux
python3 -m src.static_feature_extractor

# Windows
python -m src.static_feature_extractor
```

This regenerates `data/flakeflagger/static_features.csv`.

---

## Results

### Baseline comparison — all evaluated on FlakeFlagger (stratified 5-fold CV)

| Approach | F1 | Precision | Recall | Requires runtime? |
|---|---|---|---|---|
| Smell-only (FlakeFlagger smells) | 0.196 | 0.135 | 0.358 | No |
| History-only (hIndex commit features) | 0.480 | 0.670 | 0.374 | Yes (commit history) |
| **Our approach — static features** | **0.667** | **0.747** | **0.602** | **No** |
| Full FlakeFlagger (smell + history + runtime) | 0.727 | 0.784 | 0.678 | Yes (full instrumentation) |

Our static approach outperforms the smell-only baseline by **+0.471 F1** and the history-only baseline by **+0.187 F1**, while sitting only **0.060 below** the fully-instrumented ceiling — with zero test reruns and zero runtime instrumentation.

### Our approach — Static features detail (12,699 tests, 269 flaky, 2.12%)

Best model: LightGBM + SMOTE + threshold optimization — stratified 5-fold CV:

| Model | Avg F1 | Precision (flaky) | Recall (flaky) |
|---|---|---|---|
| Random Forest | 0.629 | 0.755 | 0.539 |
| XGBoost | 0.665 | 0.761 | 0.591 |
| LightGBM (best) | **0.667** | **0.747** | **0.602** |

### Cross-project generalization — Leave-One-Project-Out (LOPO)

Trained on all projects, tested on each held-out project (15 projects with at least one flaky test):

| Model | Avg F1 | Avg Precision | Avg Recall |
|---|---|---|---|
| Random Forest | 0.001 | 0.001 | 0.001 |
| XGBoost | 0.051 | 0.124 | 0.051 |

Cross-project generalization is very low, consistent with findings in the flaky test literature: flakiness patterns are highly project-specific. The static features that distinguish flaky from non-flaky tests within one project do not reliably transfer to unseen projects. This is a known open problem — per-project training (as in the within-project CV above) is substantially more effective.

---

## Static Features Extracted (46 total)

### Core structural features
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

### Density / complexity features
| Feature | Description |
|---|---|
| `assert_density` | Assertions per test method |
| `loc_per_test` | Lines of code per test method |
| `has_timeout_annotation` | Binary: `@Test(timeout=...)` present |
| `timeout_count` | Occurrences of timeout-related keywords |
| `polling_count` | Count of polling/awaiting primitives (Awaitility, etc.) |

### Environment / dependency features
| Feature | Description |
|---|---|
| `has_env_access` | Binary: `System.getenv` / `System.getProperty` access |
| `has_db_access` | Binary: JDBC, Hibernate, or `@Transactional` present |
| `has_injection` | Binary: `@Autowired`, `@Mock`, `@Inject`, etc. |
| `has_static_field` | Binary: static (non-final) field declarations |
| `thread_join_count` | Number of `.join()` calls |
| `notify_count` | Number of `notify()` / `notifyAll()` calls |
| `broad_catch_count` | Catch blocks catching `Exception` / `Throwable` |
| `file_io_count` | Count of file I/O constructor/method calls |
| `network_io_count` | Count of network I/O calls |
| `has_rule_annotation` | Binary: JUnit `@Rule` present |
| `num_inner_classes` | Number of inner class declarations |

### Import-based features
| Feature | Description |
|---|---|
| `imports_mockito` | Binary: Mockito imported |
| `imports_powermock` | Binary: PowerMock imported |
| `imports_easymock` | Binary: EasyMock imported |
| `imports_concurrent` | Binary: `java.util.concurrent` imported |
| `imports_atomic` | Binary: `java.util.concurrent.atomic` imported |
| `imports_network` | Binary: `java.net` / `okhttp3` / Apache HTTP imported |
| `imports_spring` | Binary: Spring framework imported |
| `imports_guice` | Binary: Guice DI imported |
| `imports_jdbc` | Binary: `java.sql` imported |
| `imports_jpa` | Binary: JPA/Jakarta Persistence imported |
| `imports_nio` | Binary: `java.nio` imported |
| `imports_io` | Binary: `java.io` imported |
| `imports_awaitility` | Binary: Awaitility imported |
| `num_imports` | Total import statement count |
