def dataset_stats(data):
    total = len(data)
    flaky = sum(1 for d in data if d['label'] == 1)
    non_flaky = total - flaky

    print("\n--- Dataset Statistics ---")
    print(f"Total tests: {total}")
    print(f"Flaky tests: {flaky}")
    print(f"Non-flaky tests: {non_flaky}")

    if total > 0:
        print(f"Flaky percentage: {flaky / total:.2%}")


def check_imbalance(data):
    total = len(data)
    flaky = sum(1 for d in data if d['label'] == 1)

    if total == 0:
        print("Dataset is empty")
        return

    ratio = flaky / total

    print("\n--- Class Balance ---")
    print(f"Flaky ratio: {ratio:.2%}")

    if ratio < 0.2:
        print("Dataset is imbalanced")
    else:
        print("Dataset is relatively balanced")


def print_sample(data, n=3):
    print("\n--- Sample Data ---")

    for d in data[:n]:
        print({
            "test_name": d.get("test_name", "N/A"),
            "label": d["label"]
        })