from io import BytesIO
import os
import base64
import numpy as np
from flask import Flask, send_file, request, abort, jsonify
from flask_cors import CORS
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler
)
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Loading ControlNet Union SDXL -> {DEVICE}')
print('⚠ This will use significant VRAM (~7-8GB). Loading with maximum optimizations...\n')

# Load ControlNet Union model
controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-union-sdxl-1.0",
    torch_dtype=torch.float16
)

# Load fixed VAE for SDXL
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

# ControlNet-enabled SDXL
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✓ xformers enabled")
except:
    print("✓ Using PyTorch SDPA")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Base SDXL (no ControlNet)
pipe_base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe_base.enable_model_cpu_offload()
pipe_base.enable_attention_slicing(1)
pipe_base.enable_vae_slicing()
try:
    pipe_base.enable_xformers_memory_efficient_attention()
except:
    pass
pipe_base.scheduler = UniPCMultistepScheduler.from_config(pipe_base.scheduler.config)

print("✓ Both pipelines loaded successfully!\n")

# ControlNet modes
CONTROL_TYPES = {
    'canny': 0,
    'tile': 1,
    'depth': 2,
    'blur': 3,
    'pose': 4,
    'gray': 5,
    'low_quality': 6
}


def process_control_image(image, control_type='canny'):
    """Process input image based on control type"""
    img_array = np.array(image)

    if control_type == 'canny':
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges)
    elif control_type == 'depth':
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(gray)
    elif control_type == 'gray':
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(gray)
    elif control_type == 'blur':
        blurred = cv2.GaussianBlur(img_array, (21, 21), 0)
        return Image.fromarray(blurred)
    else:
        return image


@app.route('/')
def index():
    """Serve web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ControlNet Union SDXL</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 50px auto; padding: 20px; }
            .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            input, select, button, textarea { padding: 10px; margin: 10px 0; width: 100%; box-sizing: border-box; }
            textarea { height: 100px; }
            button { background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .preview { border: 2px dashed #ccc; padding: 20px; text-align: center; min-height: 200px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .results { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>🎨 ControlNet Union SDXL + Prompt-only Mode</h1>
        <p><strong>Control Types:</strong> Canny, Depth, Pose, Blur, Gray, Tile, Low Quality</p>
        
        <div class="container">
            <div>
                <h3>Input</h3>
                <label><input type="checkbox" id="promptOnly"> Prompt-only (no image)</label>
                <input type="file" id="imageFile" accept="image/*">
                <div id="preview" class="preview">Upload an image (or check prompt-only)</div>
                
                <select id="controlType">
                    <option value="canny">Canny Edge</option>
                    <option value="depth">Depth</option>
                    <option value="gray">Grayscale</option>
                    <option value="blur">Blur</option>
                    <option value="pose">Pose</option>
                    <option value="tile">Tile</option>
                </select>
                
                <textarea id="prompt" placeholder="Enter your prompt...">masterpiece, best quality, highres</textarea>
                <textarea id="negPrompt" placeholder="Negative prompt (optional)">low quality, bad quality, sketches</textarea>
                
                <button onclick="generate()">Generate Image</button>
            </div>
            
            <div>
                <h3>Output</h3>
                <div id="result" class="preview">Generated image will appear here</div>
            </div>
        </div>
        
        <script>
            let imageData = null;
            
            document.getElementById('imageFile').addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        imageData = e.target.result.split(',')[1];
                        document.getElementById('preview').innerHTML = 
                            `<img src="${e.target.result}" alt="Input">`;
                    };
                    reader.readAsDataURL(file);
                }
            });

            document.getElementById('promptOnly').addEventListener('change', (e) => {
                if (e.target.checked) {
                    imageData = null;
                    document.getElementById('preview').innerHTML = "Prompt-only mode selected";
                }
            });
            
            async function generate() {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Generating... (30-60 seconds on laptop GPU)</p>';
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image: document.getElementById('promptOnly').checked ? null : imageData,
                            prompt: document.getElementById('prompt').value,
                            negative_prompt: document.getElementById('negPrompt').value,
                            control_type: document.getElementById('controlType').value
                        })
                    });
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = URL.createObjectURL(blob);
                        resultDiv.innerHTML = `<img src="${url}" alt="Generated">`;
                    } else {
                        const error = await response.text();
                        resultDiv.innerHTML = `<p style="color: red;">Error: ${error}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """


@app.route('/generate', methods=['POST'])
def generate():
    """Generate image with ControlNet if image provided, else plain SDXL"""
    data = request.get_json()

    if not data or 'prompt' not in data:
        abort(400, 'Missing prompt')

    prompt = data['prompt']
    negative_prompt = data.get('negative_prompt', 'low quality, bad quality')
    num_steps = data.get('steps', 20)
    guidance_scale = data.get('guidance_scale', 7.5)

    try:
        if data.get('image'):
            # ---------- CONTROLNET MODE ----------
            image_bytes = base64.b64decode(data['image'])
            input_image = Image.open(BytesIO(image_bytes)).convert('RGB')

            max_size = 1024
            if max(input_image.size) > max_size:
                input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            w, h = input_image.size
            w = (w // 8) * 8
            h = (h // 8) * 8
            input_image = input_image.resize((w, h))

            control_type = data.get('control_type', 'canny')
            controlnet_scale = data.get('controlnet_scale', 0.8)

            control_image = process_control_image(input_image, control_type)

            print(f"Generating WITH ControlNet ({control_type}): '{prompt}' ({w}x{h})")
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                width=w,
                height=h
            )
        else:
            # ---------- BASE SDXL MODE ----------
            w, h = 1024, 1024
            print(f"Generating BASE SDXL: '{prompt}' ({w}x{h})")
            result = pipe_base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=w,
                height=h
            )

        output_image = result.images[0]

        img_io = BytesIO()
        output_image.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except torch.cuda.OutOfMemoryError:
        abort(500, 'Out of VRAM. Try smaller image or restart the server.')
    except Exception as e:
        abort(500, str(e))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'xinsir/controlnet-union-sdxl-1.0',
        'device': DEVICE,
        'control_types': list(CONTROL_TYPES.keys())
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ControlNet Union SDXL Server (with Prompt-only): http://localhost:5000")
    print("="*60 + "\n")
    print("⚠ IMPORTANT: This uses 7-8GB VRAM. Close other GPU applications!")
    print("💡 Keep images under 1024px for best performance\n")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
