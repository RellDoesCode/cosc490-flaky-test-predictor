"""
AST-based feature extractor for Java test files.

Uses javalang to parse Java source into a real AST and extracts structural
signals that regex cannot capture: cyclomatic complexity, nesting depth,
shared state (instance/static fields), reflection usage, lambda density, etc.

Produces data/flakeflagger/ast_features.csv, then merges with
static_features.csv into data/flakeflagger/combined_features.csv.

Run from repo root:
    python -m src.ast_feature_extractor
"""

import os
import re
import pandas as pd
import javalang
from javalang import tree as jt

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

# AST node types that count as decision points for cyclomatic complexity
DECISION_TYPES = (
    jt.IfStatement, jt.ForStatement, jt.WhileStatement, jt.DoStatement,
    jt.SwitchStatement, jt.SwitchStatementCase, jt.CatchClause,
    jt.TernaryExpression, jt.BinaryOperation,
)


def _nesting_depth(node, depth=0):
    """Recursively compute max control-flow nesting depth."""
    BLOCK_TYPES = (
        jt.IfStatement, jt.ForStatement, jt.WhileStatement,
        jt.DoStatement, jt.TryStatement, jt.SynchronizedStatement,
    )
    local_max = depth
    children = []
    if hasattr(node, 'children'):
        for child in node.children:
            if child is None:
                continue
            if isinstance(child, list):
                children.extend(child)
            else:
                children.append(child)

    for child in children:
        if not isinstance(child, jt.Node):
            continue
        new_depth = depth + (1 if isinstance(child, BLOCK_TYPES) else 0)
        local_max = max(local_max, _nesting_depth(child, new_depth))
    return local_max


def extract_ast_features(java_source):
    """Return a dict of AST features, or None if the file fails to parse."""
    try:
        ast = javalang.parse.parse(java_source)
    except Exception:
        return None

    # ── Collect all nodes ────────────────────────────────────────────────────
    all_nodes = list(ast)  # list of (path, node) tuples
    nodes_only = [n for _, n in all_nodes]

    # ── Class-level structure ────────────────────────────────────────────────
    classes = [n for n in nodes_only if isinstance(n, jt.ClassDeclaration)]
    methods = [n for n in nodes_only if isinstance(n, jt.MethodDeclaration)]
    fields  = [n for n in nodes_only if isinstance(n, jt.FieldDeclaration)]

    # Instance vs static field counts
    static_fields   = [f for f in fields if 'static' in (f.modifiers or [])]
    instance_fields = [f for f in fields if 'static' not in (f.modifiers or [])]

    # ── Method-level metrics ─────────────────────────────────────────────────
    test_methods = [m for m in methods
                    if any(str(a).startswith('Test') or 'Test' in str(a)
                           for a in (m.annotations or []))]

    # Cyclomatic complexity: count decision nodes across all test methods
    decision_nodes = [n for n in nodes_only if isinstance(n, DECISION_TYPES)]
    # Also count short-circuit && and ||
    binary_ops = [n for n in nodes_only
                  if isinstance(n, jt.BinaryOperation)
                  and getattr(n, 'operator', '') in ('&&', '||')]
    cyclomatic = len(decision_nodes) + len(binary_ops) + max(1, len(test_methods))

    # Max nesting depth (expensive — compute per test method, cap at first 10)
    max_depth = 0
    for m in test_methods[:10]:
        try:
            max_depth = max(max_depth, _nesting_depth(m))
        except Exception:
            pass

    # Statements per test method (average)
    stmt_counts = []
    for m in test_methods:
        body = m.body or []
        stmt_counts.append(len(body))
    avg_stmts_per_test = round(sum(stmt_counts) / len(stmt_counts), 4) if stmt_counts else 0.0
    max_stmts_per_test = max(stmt_counts) if stmt_counts else 0

    # ── Invocations and instantiations ───────────────────────────────────────
    invocations = [n for n in nodes_only if isinstance(n, jt.MethodInvocation)]
    creators    = [n for n in nodes_only if isinstance(n, jt.ClassCreator)]

    # Unique types instantiated
    unique_types = len({getattr(c.type, 'name', '') for c in creators if c.type})

    # ── Lambda / anonymous class density ────────────────────────────────────
    lambdas    = [n for n in nodes_only if isinstance(n, jt.LambdaExpression)]
    anon_class = [n for n in nodes_only if isinstance(n, jt.ClassCreator)
                  and n.body is not None]

    # ── Exception handling ───────────────────────────────────────────────────
    catches = [n for n in nodes_only if isinstance(n, jt.CatchClause)]
    catch_types = set()
    for c in catches:
        if c.parameter and c.parameter.types:
            catch_types.update(str(t) for t in c.parameter.types)
    num_distinct_exception_types = len(catch_types)

    broad_exceptions = {'Exception', 'Throwable', 'RuntimeException', 'Error'}
    has_broad_catch = int(bool(catch_types & broad_exceptions))

    # ── Reflection usage ─────────────────────────────────────────────────────
    reflection_patterns = re.compile(
        r'\b(Class\.forName|getMethod|getDeclaredMethod|invoke|'
        r'getDeclaredField|setAccessible|newInstance)\s*\('
    )
    has_reflection = int(bool(reflection_patterns.search(java_source)))

    # ── Literals ─────────────────────────────────────────────────────────────
    literals = [n for n in nodes_only if isinstance(n, jt.Literal)]
    string_literals = [n for n in literals
                       if isinstance(getattr(n, 'value', None), str)
                       and n.value.startswith('"')]

    # ── Shared state between test methods ───────────────────────────────────
    # Instance fields = state that persists across test method calls if setUp
    # doesn't reset them — a known flakiness source
    num_instance_fields = sum(len(f.declarators) for f in instance_fields)
    num_static_fields   = sum(len(f.declarators) for f in static_fields)

    # ── Casts (unsafe downcasting = potential ClassCastException) ────────────
    casts = [n for n in nodes_only if isinstance(n, jt.Cast)]

    # ── Synchronization primitives ───────────────────────────────────────────
    synced = [n for n in nodes_only if isinstance(n, jt.SynchronizedStatement)]

    # ── Return value ────────────────────────────────────────────────────────
    return {
        'ast_cyclomatic_complexity':    cyclomatic,
        'ast_max_nesting_depth':        max_depth,
        'ast_num_methods':              len(methods),
        'ast_num_test_methods':         len(test_methods),
        'ast_num_instance_fields':      num_instance_fields,
        'ast_num_static_fields':        num_static_fields,
        'ast_num_invocations':          len(invocations),
        'ast_num_creators':             len(creators),
        'ast_unique_types_created':     unique_types,
        'ast_num_lambdas':              len(lambdas),
        'ast_num_anon_classes':         len(anon_class),
        'ast_num_catches':              len(catches),
        'ast_distinct_exception_types': num_distinct_exception_types,
        'ast_has_broad_catch':          has_broad_catch,
        'ast_has_reflection':           has_reflection,
        'ast_num_string_literals':      len(string_literals),
        'ast_num_casts':                len(casts),
        'ast_num_synchronized':         len(synced),
        'ast_avg_stmts_per_test':       avg_stmts_per_test,
        'ast_max_stmts_per_test':       max_stmts_per_test,
        'ast_num_classes':              len(classes),
        'ast_binary_logic_ops':         len(binary_ops),
    }


def find_java_files_ast(project_dir):
    """Walk project dir, yield (class_key, ast_features) for parseable .java files."""
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

            features = extract_ast_features(source)
            if features is None:
                continue

            # Derive class_key same way as static_feature_extractor
            pkg_match = re.search(r'^\s*package\s+([\w.]+)\s*;', source, re.MULTILINE)
            package   = pkg_match.group(1).lower() if pkg_match else ''
            class_name = fname.replace('.java', '').lower()
            class_key  = f"{package}.{class_name}" if package else class_name

            yield class_key, features


def build_ast_features(test_files_dir, labeled_csv):
    df_labels = pd.read_csv(labeled_csv)
    if 'flaky' in df_labels.columns:
        df_labels = df_labels.rename(columns={'flaky': 'label'})

    print("Building test_name index...")
    class_key_to_rows = {}
    for idx, row in df_labels.iterrows():
        test_name = str(row['test_name']).lower()
        parts = test_name.rsplit('.', 1)
        if len(parts) == 2:
            class_key_to_rows.setdefault(parts[0], []).append(idx)

    all_records = []
    total_matched = 0
    parse_errors = 0
    total_files = 0

    for project_csv, folder in PROJECT_FOLDER_MAP.items():
        project_dir = os.path.join(test_files_dir, folder)
        if not os.path.isdir(project_dir):
            print(f"  [MISSING] {folder}")
            continue

        project_rows = df_labels[df_labels['project'] == project_csv]
        matched = 0

        for class_key, features in find_java_files_ast(project_dir):
            total_files += 1
            if class_key not in class_key_to_rows:
                continue
            for idx in class_key_to_rows[class_key]:
                if df_labels.at[idx, 'project'] != project_csv:
                    continue
                row_data = {
                    'project':   project_csv,
                    'test_name': df_labels.at[idx, 'test_name'],
                    'label':     df_labels.at[idx, 'label'],
                }
                row_data.update(features)
                all_records.append(row_data)
                matched += 1

        total_matched += matched
        print(f"  {project_csv}: {matched}/{len(project_rows)} tests matched")

    print(f"\nTotal: {total_matched} labeled tests matched from {total_files} Java files")
    return pd.DataFrame(all_records)


if __name__ == '__main__':
    TEST_FILES_DIR  = 'data/flakeflagger/test_files'
    LABELED_CSV     = 'data/flakeflagger/processed_data.csv'
    AST_OUTPUT      = 'data/flakeflagger/ast_features.csv'
    STATIC_CSV      = 'data/flakeflagger/static_features.csv'
    COMBINED_OUTPUT = 'data/flakeflagger/combined_features.csv'

    print("Extracting AST features...")
    df_ast = build_ast_features(TEST_FILES_DIR, LABELED_CSV)

    if df_ast.empty:
        print("No AST matches found.")
    else:
        df_ast.to_csv(AST_OUTPUT, index=False)
        print(f"\nSaved {len(df_ast)} rows to {AST_OUTPUT}")
        print("Label distribution:")
        print(df_ast['label'].value_counts())

        # Merge with static features
        print("\nMerging with static features...")
        df_static = pd.read_csv(STATIC_CSV)
        df_combined = df_static.merge(
            df_ast.drop(columns=['label']),
            on=['project', 'test_name'],
            how='inner'
        )
        df_combined.to_csv(COMBINED_OUTPUT, index=False)
        print(f"Combined dataset: {len(df_combined)} rows, {df_combined.shape[1]} columns")
        print(f"  Flaky: {df_combined['label'].sum()} ({df_combined['label'].mean()*100:.2f}%)")
        print(f"  AST features: {[c for c in df_combined.columns if c.startswith('ast_')]}")
