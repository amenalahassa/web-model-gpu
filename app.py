import gradio as gr

# import shutil
# from pathlib import Path
# current_dir = Path(__file__).parent

header = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.11.2/css/all.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome-animation/0.2.1/font-awesome-animation.min.css">
"""

css = """

body
{
  background: #000e29;
}

.alert>.start-icon {
    margin-right: 0;
    min-width: 20px;
    text-align: center;
}

.alert>.start-icon {
    margin-right: 5px;
}

.greencross
{
  font-size:18px;
      color: #25ff0b;
    text-shadow: none;
}

.alert-simple.alert-success
{
  border: 1px solid rgba(36, 241, 6, 0.46);
    background-color: rgba(7, 149, 66, 0.12156862745098039);
    box-shadow: 0px 0px 2px #259c08;
    color: #0ad406;
  text-shadow: 2px 1px #00040a;
  transition:0.5s;
  cursor:pointer;
}
.alert-success:hover{
  background-color: rgba(7, 149, 66, 0.35);
  transition:0.5s;
}
.alert-simple.alert-info
{
  border: 1px solid rgba(6, 44, 241, 0.46);
    background-color: rgba(7, 73, 149, 0.12156862745098039);
    box-shadow: 0px 0px 2px #0396ff;
    color: #0396ff;
  text-shadow: 2px 1px #00040a;
  transition:0.5s;
  cursor:pointer;
}

.alert-info:hover
{
  background-color: rgba(7, 73, 149, 0.35);
  transition:0.5s;
}

.blue-cross
{
  font-size: 18px;
    color: #0bd2ff;
    text-shadow: none;
}

.alert-simple.alert-warning
{
      border: 1px solid rgba(241, 142, 6, 0.81);
    background-color: rgba(220, 128, 1, 0.16);
    box-shadow: 0px 0px 2px #ffb103;
    color: #ffb103;
    text-shadow: 2px 1px #00040a;
  transition:0.5s;
  cursor:pointer;
}

.alert-warning:hover{
  background-color: rgba(220, 128, 1, 0.33);
  transition:0.5s;
}

.warning
{
      font-size: 18px;
    color: #ffb40b;
    text-shadow: none;
}

.alert-simple.alert-danger
{
  border: 1px solid rgba(241, 6, 6, 0.81);
    background-color: rgba(220, 17, 1, 0.16);
    box-shadow: 0px 0px 2px #ff0303;
    color: #ff0303;
    text-shadow: 2px 1px #00040a;
  transition:0.5s;
  cursor:pointer;
}

.alert-danger:hover
{
     background-color: rgba(220, 17, 1, 0.33);
  transition:0.5s;
}

.danger
{
      font-size: 18px;
    color: #ff0303;
    text-shadow: none;
}

.alert-simple.alert-primary
{
  border: 1px solid rgba(6, 241, 226, 0.81);
    background-color: rgba(1, 204, 220, 0.16);
    box-shadow: 0px 0px 2px #03fff5;
    color: #03d0ff;
    text-shadow: 2px 1px #00040a;
  transition:0.5s;
  cursor:pointer;
}

.alert-primary:hover{
  background-color: rgba(1, 204, 220, 0.33);
   transition:0.5s;
}

.alertprimary
{
      font-size: 18px;
    color: #03d0ff;
    text-shadow: none;
}

.square_box {
    position: absolute;
    -webkit-transform: rotate(45deg);
    -ms-transform: rotate(45deg);
    transform: rotate(45deg);
    border-top-left-radius: 45px;
    opacity: 0.302;
}

.square_box.box_three {
    background-image: -moz-linear-gradient(-90deg, #290a59 0%, #3d57f4 100%);
    background-image: -webkit-linear-gradient(-90deg, #290a59 0%, #3d57f4 100%);
    background-image: -ms-linear-gradient(-90deg, #290a59 0%, #3d57f4 100%);
    opacity: 0.059;
    left: -80px;
    top: -60px;
    width: 500px;
    height: 500px;
    border-radius: 45px;
}

.square_box.box_four {
    background-image: -moz-linear-gradient(-90deg, #290a59 0%, #3d57f4 100%);
    background-image: -webkit-linear-gradient(-90deg, #290a59 0%, #3d57f4 100%);
    background-image: -ms-linear-gradient(-90deg, #290a59 0%, #3d57f4 100%);
    opacity: 0.059;
    left: 150px;
    top: -25px;
    width: 550px;
    height: 550px;
    border-radius: 45px;
}

.alert:before {
    content: '';
    position: absolute;
    width: 0;
    height: calc(100% - 44px);
    border-left: 1px solid;
    border-right: 2px solid;
    border-bottom-right-radius: 3px;
    border-top-right-radius: 3px;
    left: 0;
    top: 50%;
    transform: translate(0,-50%);
      height: 20px;
}

.fa-times
{
-webkit-animation: blink-1 2s infinite both;
	        animation: blink-1 2s infinite both;
}


/**
 * ----------------------------------------
 * animation blink-1
 * ----------------------------------------
 */
@-webkit-keyframes blink-1 {
  0%,
  50%,
  100% {
    opacity: 1;
  }
  25%,
  75% {
    opacity: 0;
  }
}
@keyframes blink-1 {
  0%,
  50%,
  100% {
    opacity: 1;
  }
  25%,
  75% {
    opacity: 0;
  }
}

/**
Custom CSS for Gradio
*/

"""

info_alert_text = """
<div class="alert fade alert-simple alert-info alert-dismissible text-left font__family-montserrat font__size-16 font__weight-light brk-library-rendered rendered show" role="alert" data-brk-library="component__alert">
  <i class="start-icon  fa fa-info-circle faa-shake animated"></i>
  <strong class="font__weight-semibold">Heads up!</strong> 
  <p class="font__weight-light">
  The GPU memory usage estimation above only show how much memory the model will take on the GPU. It's not the actual memory usage needed to train the model or use it for inference.
  You can find more information <a href="https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management" target="_blank">here</a>.
  </p>
</div>
"""

error_alert_text = """
<div class="alert fade alert-simple alert-danger alert-dismissible text-left font__family-montserrat font__size-16 font__weight-light brk-library-rendered rendered show" role="alert" data-brk-library="component__alert">
  <i class="start-icon far fa-times-circle faa-pulse animated"></i>
  <strong class="font__weight-semibold">Warning</strong> 
  <p class="font__weight-light">{error}</p>  
</div>
"""

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
    return [f"{memory} GB", True if memory > 0 else False]

# def delete_directory(req: gr.Request):
#     if not req.username:
#         return
#     user_dir: Path = current_dir / req.username
#     shutil.rmtree(str(user_dir))

with gr.Blocks(head=header, css=css, delete_cache=(43200,43200)) as demo:
    model_precision = gr.State("fp32")
    model_source = gr.State("import")
    uploaded_file = gr.State(None)
    hf_model_name = gr.State()
    msg_error = gr.State()
    supported_file_types = ["pt", "pth", "h5", "hdf5", "onnx"]
    has_computed_gpu_memory = gr.State(False)

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

                num_params = gr.Number(label="Number of Parameters")
                gpu_memory = gr.Textbox(label="GPU memory expressed in Gigabyte(GB)", show_copy_button=True)
                num_params.change(compute_gpu_memory, inputs=[num_params, model_precision], outputs=[gpu_memory, has_computed_gpu_memory])

                if runtime_error:
                    gr.HTML(error_alert_text.format(error=runtime_error))

                info = gr.HTML(info_alert_text, visible=False)
                gpu_memory.change(fn=lambda x: gr.HTML(info_alert_text, visible=True) if x != "0.0 GB" else gr.HTML(info_alert_text, visible=False), inputs=gpu_memory, outputs=info)

        def compute_model_size(input_source, input_precision, input_file, input_hf_model):
            if input_source == "import":
                model_size, error = get_model_size_from_checkpoint(input_file, input_precision)
            else:
                model_size, error = get_model_size_from_hf(input_hf_model, input_precision)

            return [model_size, error]

        compute_btn.click(compute_model_size, inputs=[model_source, model_precision, uploaded_file, hf_model_name], outputs=[num_params, msg_error])

    # demo.unload(delete_directory)

demo.launch()