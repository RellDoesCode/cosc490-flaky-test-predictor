def remove_unlabeled(data):
    return [d for d in data if d['label'] is not None]


def remove_duplicates(data):
    seen = set()
    unique_data = []

    for d in data:
        identifier = tuple(sorted(d.items()))
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(d)

    return unique_data


def standardize_labels(data):
    for d in data:
        label = str(d['label']).lower()

        if label in ['1', 'true', 'flaky']:
            d['label'] = 1
        else:
            d['label'] = 0

    return data


def clean_dataset(data):
    print("Cleaning dataset...")

    original_size = len(data)

    data = remove_unlabeled(data)
    print(f"After removing unlabeled: {len(data)}")

    data = remove_duplicates(data)
    print(f"After removing duplicates: {len(data)}")

    data = standardize_labels(data)

    print(f"Total removed: {original_size - len(data)}")

    return data