def load_model(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)

def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    return loss, accuracy

def main(model_path, test_data, test_labels):
    model = load_model(model_path)
    loss, accuracy = evaluate_model(model, test_data, test_labels)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Evaluate the trained CIFAR-10 model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data (numpy array).')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to the test labels (numpy array).')

    args = parser.parse_args()

    test_data = np.load(args.test_data)
    test_labels = np.load(args.test_labels)

    main(args.model_path, test_data, test_labels)