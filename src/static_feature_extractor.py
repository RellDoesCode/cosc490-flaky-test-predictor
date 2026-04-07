import os
import re
import pandas as pd

# Maps project name in CSV -> folder name in test_files/
PROJECT_FOLDER_MAP = {
    'logback':          'qos-ch-logback',
    'incubator-dubbo':  'apache-incubator-dubbo',
    'orbit':            'orbit-orbit',
    'handlebars.java':  'jknack-handlebars',
    'http-request':     'kevinsawicki-http-request',
    'jimfs':            'google-jimfs',
    'zxing':            'zxing-zxing',
    'hector':           'hector-client-hector',
    'okhttp':           'square-okhttp',
    'ninja':            'ninjaframework-ninja',
    'achilles':         'doanduyhai-Achilles',
    'elastic-job-lite': 'elasticjob-elastic-job-lite',
    'undertow':         'undertow-io-undertow',
    'activiti':         'activiti-activiti',
    'ambari':           'apache-ambari',
    'commons-exec':     'apache-commons-exec',
    'hbase':            'apache-hbase',
    'httpcore':         'apache-httpcore',
    'assertj-core':     'joel-costigliola-assertj-core',
    'java-websocket':   'tootallnate-java-websocket',
    'wildfly':          'wildfly-wildfly',
    'spring-boot':      'spring-projects-spring-boot',
    'wro4j':            'wro4j-wro4j',
    'alluxio':          'Alluxio-alluxio',
}


def extract_features(java_source):
    """Extract static features from a Java test file's source text."""
    lines = java_source.splitlines()
    non_blank_lines = [l for l in lines if l.strip()]

    # Package declaration
    pkg_match = re.search(r'^\s*package\s+([\w.]+)\s*;', java_source, re.MULTILINE)
    package = pkg_match.group(1).lower() if pkg_match else ''

    return {
        'loc':                  len(non_blank_lines),
        'num_asserts':          len(re.findall(r'\bassert\w*\s*\(', java_source)),
        'thread_sleep_count':   len(re.findall(r'Thread\.sleep\s*\(', java_source)),
        'has_thread_sleep':     int(bool(re.search(r'Thread\.sleep\s*\(', java_source))),
        'async_wait_count':     len(re.findall(r'\b(await|wait|CountDownLatch|CyclicBarrier|Semaphore)\b', java_source)),
        'has_async_wait':       int(bool(re.search(r'\b(await|wait|CountDownLatch|CyclicBarrier|Semaphore)\b', java_source))),
        'has_file_io':          int(bool(re.search(r'\b(File|FileInputStream|FileOutputStream|Files|FileWriter|FileReader|Path)\b', java_source))),
        'has_network_io':       int(bool(re.search(r'\b(Socket|ServerSocket|URL|HttpURLConnection|HttpClient|OkHttpClient)\b', java_source))),
        'has_concurrency':      int(bool(re.search(r'\b(Thread|ExecutorService|Executor|Future|Runnable|Callable|synchronized)\b', java_source))),
        'num_test_methods':     len(re.findall(r'@Test\b', java_source)),
        'num_try_catch':        len(re.findall(r'\bcatch\s*\(', java_source)),
        'has_setup_teardown':   int(bool(re.search(r'@(Before|After|BeforeClass|AfterClass|BeforeEach|AfterEach|BeforeAll|AfterAll)\b', java_source))),
        'num_conditionals':     len(re.findall(r'\b(if|for|while|switch)\s*\(', java_source)),
        'has_random':           int(bool(re.search(r'\b(Random|Math\.random)\b', java_source))),
        'has_system_time':      int(bool(re.search(r'\b(System\.currentTimeMillis|System\.nanoTime|new Date\(\)|LocalDateTime|Instant\.now)\b', java_source))),
        'num_annotations':      len(re.findall(r'(?<!import\s)@[A-Z]\w+', java_source)),
        '_package':             package,
    }


def find_java_files(project_dir):
    """Walk a project directory and yield (file_path, class_key) for .java files."""
    for root, _, files in os.walk(project_dir):
        for fname in files:
            if not fname.endswith('.java'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding='utf-8', errors='ignore') as f:
                    source = f.read()
            except Exception:
                continue

            features = extract_features(source)
            class_name = fname.replace('.java', '').lower()
            package = features.pop('_package')

            if package:
                class_key = f"{package}.{class_name}"
            else:
                class_key = class_name

            yield class_key, features


def build_static_features(test_files_dir, labeled_csv):
    df_labels = pd.read_csv(labeled_csv)
    if 'flaky' in df_labels.columns:
        df_labels = df_labels.rename(columns={'flaky': 'label'})

    # Build a lookup: class_key -> list of row indices in df_labels
    # test_name format: "package.classname.methodname" (all lowercase)
    # class_key:        "package.classname"
    # So test_name.startswith(class_key + ".") is our match condition
    print("Building test_name index...")
    class_key_to_rows = {}
    for idx, row in df_labels.iterrows():
        test_name = str(row['test_name']).lower()
        parts = test_name.rsplit('.', 1)        # split off method name
        if len(parts) == 2:
            key = parts[0]
            class_key_to_rows.setdefault(key, []).append(idx)

    all_records = []
    total_matched = 0
    total_files = 0

    for project_csv, folder in PROJECT_FOLDER_MAP.items():
        project_dir = os.path.join(test_files_dir, folder)
        if not os.path.isdir(project_dir):
            print(f"  [MISSING] {folder}")
            continue

        project_rows = df_labels[df_labels['project'] == project_csv]
        matched = 0

        for class_key, features in find_java_files(project_dir):
            total_files += 1
            if class_key not in class_key_to_rows:
                continue
            for idx in class_key_to_rows[class_key]:
                if df_labels.at[idx, 'project'] != project_csv:
                    continue
                row_data = {'project': project_csv,
                            'test_name': df_labels.at[idx, 'test_name'],
                            'label': df_labels.at[idx, 'label']}
                row_data.update(features)
                all_records.append(row_data)
                matched += 1

        total_matched += matched
        print(f"  {project_csv}: {matched}/{len(project_rows)} tests matched")

    print(f"\nTotal: {total_matched} labeled tests matched from {total_files} Java files")
    return pd.DataFrame(all_records)


if __name__ == '__main__':
    TEST_FILES_DIR = 'data/flakeflagger/test_files'
    LABELED_CSV    = 'data/flakeflagger/processed_data.csv'
    OUTPUT_CSV     = 'data/flakeflagger/static_features.csv'

    df = build_static_features(TEST_FILES_DIR, LABELED_CSV)

    if df.empty:
        print("No matches found — check folder mapping or test_name format.")
    else:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        print("\nFeature columns:", [c for c in df.columns if c not in ('project', 'test_name', 'label')])
