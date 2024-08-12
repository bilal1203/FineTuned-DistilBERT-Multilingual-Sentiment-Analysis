from datasets import load_dataset

def load_and_preprocess_data():
    # Load the dataset
    dataset = load_dataset("mteb/amazon_reviews_multi", trust_remote_code=True)

    # Basic preprocessing
    def preprocess_function(examples):
        return {
            'text': examples['review'],
            'label': examples['stars'] - 1  # Assuming 1-5 star rating, convert to 0-4
        }

    preprocessed_dataset = dataset.map(preprocess_function, remove_columns=dataset['train'].column_names)

    # Split the dataset
    train_val = preprocessed_dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_val['train']
    val_dataset = train_val['test']
    test_dataset = preprocessed_dataset['test']

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    load_and_preprocess_data()