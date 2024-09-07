import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer, TFAutoModel
import onnx
from torch.onnx import export

def save_model(model_name, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for models
    model = None
    tf_model = None

    # Load the PyTorch model
    try:
        model = AutoModel.from_pretrained(model_name)
        print("PyTorch model loaded successfully")
    except Exception as e:
        print(f"Failed to load PyTorch model: {e}")

    # Load the TensorFlow model
    try:
        tf_model = TFAutoModel.from_pretrained(model_name)
        print("TensorFlow model loaded successfully")
    except Exception as e:
        print(f"Failed to load TensorFlow model: {e}")

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return  # Exit if tokenizer loading fails, as further steps depend on it

    # Example input for the model
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids

    if model:
        # 1. Save in 'pt' and 'pth' formats (PyTorch)
        try:
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            print("PyTorch model saved as model.pt")
        except Exception as e:
            print(f"Failed to save PyTorch model as .pt: {e}")

        try:
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
            print("PyTorch model saved as model.pth")
        except Exception as e:
            print(f"Failed to save PyTorch model as .pth: {e}")

        # 3. Save in 'onnx' format
        try:
            # Define a dummy input for the ONNX export
            dummy_input = torch.zeros(1, input_ids.shape[1], dtype=torch.long)
            export(model, dummy_input, os.path.join(output_dir, "model.onnx"), input_names=["input_ids"], output_names=["output"])
            print("Model saved as model.onnx")

            # Validate the ONNX model
            onnx_model = onnx.load(os.path.join(output_dir, "model.onnx"))
            onnx.checker.check_model(onnx_model)
            print("ONNX model validated successfully")
        except Exception as e:
            print(f"Failed to save or validate ONNX model: {e}")

    if tf_model:
        # 2. Save in 'h5' and 'hdf5' formats (Keras)
        try:
            tf_model.save(os.path.join(output_dir, "model.h5"))
            print("Keras model saved as model.h5")
        except Exception as e:
            print(f"Failed to save Keras model as .h5: {e}")

        try:
            tf_model.save(os.path.join(output_dir, "model.hdf5"))
            print("Keras model saved as model.hdf5")
        except Exception as e:
            print(f"Failed to save Keras model as .hdf5: {e}")

    print(f"Models saved in pt, pth, h5, hdf5, and onnx formats in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a GPT-2 model in various formats.")
    parser.add_argument("model_name", type=str, help="The name of the model to load from Hugging Face.")
    parser.add_argument("output_dir", type=str, help="The directory where the model files will be saved.")

    args = parser.parse_args()

    save_model(args.model_name, args.output_dir)
