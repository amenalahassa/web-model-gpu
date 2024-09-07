import gradio as gr

def get_model_size(model_size, precision):
    if precision == "fp16":
        model_size *= 0.5
    elif precision == "int8":
        model_size *= 0.125
    elif precision == "int4":
        model_size *= 0.0625
    return model_size

def get_model_size_from_checkpoint(file, precision):
    from pathlib import Path

    num_params = 0
    error = None
    filepath = Path(file)
    extension = filepath.suffix[1:]

    try:

        if extension in ["pth", "pt"]:
            import torch
            checkpoint = torch.load(file, weights_only=False)

            # If the checkpoint contains only the state_dict, use it directly
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Calculate the total number of parameters
            # Assuming that the model is composed of multiple children modules/models
            for child in state_dict.values():
                # Check if the parameter is a model
                if isinstance(child, torch.nn.Module):
                    # Calculate the number of parameters in the model
                    num_params += sum(p.numel() for p in child.parameters())

            # Calculate the number of parameters of direct children/layers
            for param in state_dict.values():
                # Check if the parameter has the attribute `numel`
                if hasattr(param, "numel"):
                    num_params += param.numel()

        elif extension in ["h5", "hdf5"]:
            from tensorflow.keras.models import load_model

            model = load_model(file)
            model.compile()
            # Calculate the total number of parameters
            num_params = model.count_params()

        elif extension in ["onnx"]:
            import onnx
            from onnx import numpy_helper

            model = onnx.load(file)
            num_params = sum([numpy_helper.to_array(tensor).size for tensor in model.graph.initializer])

        else:
            error = "Unsupported file format. Please upload a PyTorch/Keras/ONNX model checkpoint."

    except Exception as e:
        error = str(e)

    if num_params == 0 and error is None:
        error = "No parameters found in the model checkpoint"

    return get_model_size(num_params, precision), error

def get_model_size_from_hf(model_name, precision):
    from transformers import AutoModel
    num_params = 0
    error = None

    try:
        model = AutoModel.from_pretrained(model_name)
        num_params = sum(param.numel() for param in model.parameters())
    except Exception as e:
        error = str(e)

    return get_model_size(num_params, precision), error

def compute_gpu_memory(input_model_size, model_precision):
    P = input_model_size
    Q = 32 if model_precision == "fp32" else 16 if model_precision == "fp16" else 8 if model_precision == "int8" else 4
    memory = P * Q / 8 / 1024 / 1024 / 1024
    return f"{memory} GB"


with gr.Blocks() as demo:
    model_precision = gr.State("fp32")
    model_source = gr.State("import")
    uploaded_file = gr.State(None)
    hf_model_name = gr.State()
    msg_error = gr.State()
    supported_file_types = ["pt", "pth", "h5", "hdf5", "onnx"]

    gr.Markdown(
        """
        # Wondering how much memory your model will take?
        This app helps you estimate the memory usage of a model on GPU.
        """
    )

    checkpoint_radio = gr.Radio(
        [("Import model checkpoint", "import"), ("Use model from Hugging Face", "hf")],
        value="import",
        label="Choose a model source"
    )

    checkpoint_radio.change(fn=lambda x: x, inputs=checkpoint_radio, outputs=model_source)
    @gr.render(inputs=[model_source, msg_error])
    def rendering(source, runtime_error):

        with gr.Row():
            with gr.Column():
                if source == "import":
                    gr.Markdown("Upload a model checkpoint file. Supported formats are PyTorch, Keras, and ONNX.")
                    uploader = gr.File(label=f'Upload Model Checkpoint [{" | ".join(supported_file_types)}]', file_types=supported_file_types, file_count="single", type="filepath")
                    uploader.upload(fn=lambda x: x, inputs=uploader, outputs=uploaded_file)
                else:
                    mode_name_textbox = gr.Textbox(label="Model Name", placeholder="e.g. facebook/bart-large")
                    mode_name_textbox.change(fn=lambda x: x, inputs=mode_name_textbox, outputs=hf_model_name)

                precision_radio = gr.Radio(
                    [
                        ("FP32 (32-bit floating point)", "fp32"),
                        ("FP16 (half/BF16) (16-bit floating point)", "fp16"),
                        ("INT8 (8-bit integer)", "int8"),
                        ("INT4 (4-bit integer)", "int4"),
                    ],
                    value=model_precision.value,
                    label="Select the Precision or Size of the model parameters"
                )
                precision_radio.change(fn=lambda x: x, inputs=precision_radio, outputs=model_precision)
                compute_btn = gr.Button("Compute")

            with gr.Column():

                if runtime_error:
                    gr.Textbox(label="Error", value=runtime_error, interactive=False, lines=5)

                num_params = gr.Number(label="Number of Parameters")
                gpu_memory = gr.Textbox(label="GPU memory expressed in Gigabyte", show_copy_button=True)
                num_params.change(compute_gpu_memory, inputs=[num_params, model_precision], outputs=[gpu_memory])

        def compute_model_size(input_source, input_precision, input_file, input_hf_model):
            if input_source == "import":
                model_size, error = get_model_size_from_checkpoint(input_file, input_precision)
            else:
                model_size, error = get_model_size_from_hf(input_hf_model, input_precision)

            return [model_size, error]

        compute_btn.click(compute_model_size, inputs=[model_source, model_precision, uploaded_file, hf_model_name], outputs=[num_params, msg_error])

demo.launch()