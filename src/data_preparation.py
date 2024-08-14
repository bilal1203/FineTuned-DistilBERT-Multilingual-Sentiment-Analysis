from datasets import load_dataset, Dataset
import pandas as pd
from typing import Tuple

def load_dataset_from_hub() -> Dataset:
    try:
        return load_dataset("mteb/amazon_reviews_multi", trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

def preprocess_data(dataset: Dataset) -> Dataset:
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['text', 'label']]
    return dataset.map(lambda examples: examples, remove_columns=columns_to_remove)

def analyze_class_distribution(dataset: Dataset) -> None:
    labels = pd.Series([example['label'] for example in dataset['train']])
    class_dist = labels.value_counts()
    imbalance_ratio = class_dist.max() / class_dist.min()

    print('Class distribution:', class_dist)
    print('Imbalance ratio:', imbalance_ratio)

def split_dataset(dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def load_and_preprocess_data() -> Tuple[Dataset, Dataset, Dataset]:
    dataset = load_dataset_from_hub()
    preprocessed_dataset = preprocess_data(dataset)
    analyze_class_distribution(preprocessed_dataset)
    return split_dataset(preprocessed_dataset)

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data()