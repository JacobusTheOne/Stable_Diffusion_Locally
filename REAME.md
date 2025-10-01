<!-- README.md - HTML styled for GitHub -->

<h1 align="center">🚀 ControlNet Union SDXL — Local Server</h1>

<p align="center">
  Run Stable Diffusion XL (SDXL) locally with optional ControlNet guidance (edge / depth / pose / etc.).<br>
  Launches a small Flask UI at <strong>http://127.0.0.1:5000</strong>.
</p>

<hr>

<h2>⚙️ Quick overview</h2>

<ul>
  <li><strong>Prompt-only mode:</strong> SDXL generates from text alone.</li>
  <li><strong>ControlNet mode:</strong> Upload an image and SDXL will be guided by edges / depth / pose, etc.</li>
  <li>Designed for laptops/GPUs with ~8GB VRAM (uses CPU offload, attention slicing, VAE slicing).</li>
</ul>

<hr>

<h2>🧰 Requirements</h2>

<ul>
  <li>Python 3.10+</li>
  <li>NVIDIA GPU &amp; CUDA (recommended for speed) — instructions below for CUDA 12.1</li>
  <li>Hugging Face account + access token for some models</li>
</ul>

<hr>

<h2>🔧 Installation (Linux / macOS)</h2>

<pre><code># create &amp; activate venv
python -m venv venv
source venv/bin/activate

# install packages
pip install diffusers transformers accelerate safetensors
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# optional: specific pinned versions (if you want these exact builds)
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# optional speedups
pip install xformers

# login to huggingface (required for some models)
huggingface-cli login
</code></pre>

<h2>🔧 Installation (Windows — PowerShell)</h2>

<pre><code># create &amp; activate venv
python -m venv venv
venv\Scripts\activate

# Allow running local scripts (run once)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# install packages
pip install diffusers transformers accelerate safetensors
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# (optional pinned versions)
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# optional speedups
pip install xformers

# login to huggingface
huggingface-cli login
</code></pre>

<p><em>Notes:</em> If you don't have an NVIDIA GPU, install the CPU-only PyTorch or matching CUDA version for your system. The `--index-url` shown installs CUDA 12.1 wheels.</p>

<hr>

<h2>📁 Files</h2>

<ul>
  <li><code>server_controlnet_sdxl.py</code> — Flask server (SDXL + ControlNet). <strong>Start this file</strong>.</li>
  <li><code>README.md</code> — this file.</li>
</ul>

<hr>

<h2>▶️ Running the server</h2>

<pre><code>python server_controlnet_sdxl.py
</code></pre>

Open the UI in your browser:

👉 <a href="http://127.0.0.1:5000">http://127.0.0.1:5000</a>

<hr>

<h2>📡 API</h2>

<h3>Prompt-only (base SDXL)</h3>

<p><strong>POST</strong> <code>/generate</code></p>

<pre><code>{
  "prompt": "A fantasy monk standing on a mountain peak",
  "negative_prompt": "low quality, bad quality",
  "steps": 20,
  "guidance_scale": 7.5
}
</code></pre>

<h3>ControlNet (upload an image)</h3>

<p>Send the image as Base64 in the `image` field (or use the provided HTML UI). Example JSON body:</p>

<pre><code>{
  "image": "<base64 image>",
  "prompt": "A cyberpunk city at night",
  "control_type": "canny",        # canny | depth | gray | blur | pose | tile
  "controlnet_scale": 0.8,
  "steps": 20,
  "guidance_scale": 7.5
}
</code></pre>

<hr>

<h2>💡 Example Python client</h2>

<pre><code>import base64
import requests

# prompt-only
resp = requests.post("http://127.0.0.1:5000/generate", json={
    "prompt": "A detailed concept art of a samurai robot",
    "negative_prompt": "low quality",
    "steps": 20
})
open("out_prompt.png","wb").write(resp.content)

# with image (ControlNet): encode file to base64 and send
with open("input.png","rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

resp = requests.post("http://127.0.0.1:5000/generate", json={
    "image": b64,
    "prompt": "Turn this into a neon city scene",
    "control_type": "canny"
})
open("out_controlnet.png","wb").write(resp.content)
</code></pre>

<hr>

<h2>🛠️ Troubleshooting</h2>

<ul>
  <li><strong>Out of VRAM</strong>: reduce input image size, lower <code>steps</code>, or close other GPU apps.</li>
  <li><strong>xformers fails to install</strong>: it’s optional. The script falls back to PyTorch attention.</li>
  <li><strong>Model download blocked</strong>: ensure you run <code>huggingface-cli login</code> and have the required model access.</li>
</ul>

<hr>

<h2>📜 License &amp; Model Terms</h2>

<p>Be sure to follow the licenses/usage terms of the models you download (Hugging Face model pages). This repo does not change model licensing.</p>

<hr>

<p align="center">Made with ❤️ — run responsibly and respect model usage guidelines.</p>
