import os
import time
import json
from pathlib import Path
from vllm import LLM, SamplingParams

def get_hf_api_key() -> str:
    """Gets the user's Huggingface API key from a config file."""
    # Note: Corrected the path joining for cross-platform compatibility
    home_dir = Path.home()
    path = home_dir / "stance-detection-german-llm" / "secrets.json"
    try:
        with open(path, "r") as config_file:
            config = json.load(config_file)
        return config.get("huggingface_api_key")
    except FileNotFoundError:
        print(f"Error: secrets file not found at {path}")
        return None

# --- Set token as an environment variable for vLLM ---
hf_token = get_hf_api_key()
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
else:
    print("Hugging Face API key not found. Exiting.")
    exit()

# --- vLLM Initialization ---
start_time = time.time()

# Define the model name
model_name = "google/gemma-3-27b-it"

# vLLM handles model loading directly from the Hub.
# It will automatically download the model and distribute it across 2 GPUs.
print("Initializing LLM with vLLM...")
lm = LLM(
    model=model_name,
    tensor_parallel_size=1, # This tells vLLM to use n GPUs
    trust_remote_code=True   # Often needed for newer models
)

elapsed_time = (time.time() - start_time) / 60
print(f" LLM setup took {elapsed_time:.2f} min.")


# --- Example Inference ---
prompts = [
    "Hello, my name is",
    "The capital of Germany is",
    "What is the best thing to do in Mannheim on a Friday night?",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

# Generate text for the prompts
print(" Generating responses...")
outputs = lm.generate(prompts, sampling_params)

# Print the outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"---")
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")