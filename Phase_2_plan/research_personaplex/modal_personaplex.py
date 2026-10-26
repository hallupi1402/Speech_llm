"""Run the upstream PersonaPlex WebSocket server on a Modal GPU.

The PersonaPlex repository is cloned while Modal builds the image. Model
weights are first downloaded into a persistent Modal Volume by ``modal run``;
the WebSocket server then loads them from that cache on Modal's GPU.

Prepare the model cache with:
    uv run --with modal modal run modal_personaplex.py

Deploy the private WebSocket server with:
    uv run --with modal modal deploy modal_personaplex.py
"""

import os
import subprocess

import modal


APP_NAME = "personaplex"
MODEL_REPO = "nvidia/personaplex-7b-v1"
HF_SECRET_NAME = "personaplexHF"
MODEL_CACHE_PATH = "/cache/huggingface"
SERVER_PORT = 8998

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name(
    "personaplex-hf-cache",
    create_if_missing=True,
)

# The upstream project supports Python 3.12 and PyTorch 2.4. Install the CUDA
# wheel first; installing moshi-personaplex afterward reuses that torch build.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "git", "libopus-dev", "pkg-config")
    .pip_install(
        "torch==2.4.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/personaplex.git /opt/personaplex"
    )
    .pip_install("/opt/personaplex/moshi")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=60 * 60,
    max_containers=1,
)
def download_model_to_volume():
    """Download gated model files once and persist them in the Modal Volume."""
    os.environ["HF_HOME"] = MODEL_CACHE_PATH

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        cache_dir=os.path.join(MODEL_CACHE_PATH, "hub"),
    )
    model_cache.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={MODEL_CACHE_PATH: model_cache},
    max_containers=1,
    timeout=60 * 60,
)
@modal.web_server(
    SERVER_PORT,
    startup_timeout=20 * 60,
    requires_proxy_auth=True,
)
def serve_personaplex():
    env = os.environ.copy()
    env["HF_HOME"] = MODEL_CACHE_PATH

    subprocess.Popen(
        [
            "python",
            "-m",
            "moshi.server",
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVER_PORT),
            "--hf-repo",
            MODEL_REPO,
        ],
        cwd="/opt/personaplex/moshi",
        env=env,
    )


@app.local_entrypoint()
def prepare_model():
    """Run the one-time Hugging Face download on Modal, without a GPU."""
    download_model_to_volume.remote()
