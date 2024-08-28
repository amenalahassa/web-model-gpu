import gradio as gr

# This is the HTML and JavaScript code for your Google AdSense ad.
ad_code = """
<div class="sampled" style="text-align:center;">
  <!-- Replace this with your Google AdSense ad code -->
  <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-400121484385497" crossorigin="anonymous"></script>
</div>
"""

def greet(name):
    return "Hello " + name + "!"

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Welcome to My Gradio App!")
    name_input = gr.Textbox(label="Enter your name")
    output = gr.Textbox(label="Greeting")

    # Add the AdSense code as HTML in the interface
    gr.HTML(ad_code)

    # Button to trigger the greeting function
    greet_button = gr.Button("Greet")
    greet_button.click(fn=greet, inputs=name_input, outputs=output)

# Launch the Gradio app
demo.launch()
