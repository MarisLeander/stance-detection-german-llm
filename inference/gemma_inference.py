import os
import io
import time
import json
import vllm
import contextlib
from pathlib import Path
from vllm import LLM, SamplingParams
from inference_helper import get_formatted_prompt, get_prompt_list, get_engineering_data, insert_prediction, get_test_batch

def get_hf_api_key() -> str:
    """Gets the user's Huggingface API key from a config file."""
    home_dir = Path.home()
    path = home_dir / "stance-detection-german-llm" / "secrets.json"
    try:
        with open(path, "r") as config_file:
            config = json.load(config_file)
        return config.get("huggingface_api_key")
    except FileNotFoundError:
        print(f"Error: secrets file not found at {path}")
        return None



def run_silently(func, *args, **kwargs):
    """
    Runs a function while redirecting all stdout and stderr output
    to a black hole, effectively silencing it.

    Args:
        func: The function to run silently.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the function that was run.
    """
    # Redirect stdout and stderr to a dummy stream
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        # Call the original function
        result = func(*args, **kwargs)
    
    return result
    
def do_gemma_inference(prompt:str, input_data, llm:vllm.entrypoints.llm.LLM, sampling_params:vllm.sampling_params.SamplingParams) -> str:
    # Generate the response
    responses = run_silently(llm.generate, [prompt], sampling_params)
    # Print the output
    if len(responses) > 1:
        print("Output was longer than 1!!")
    else:
        generated_text = responses[0].outputs[0].text.strip() 
        return generated_text

def gemma_cot_predictions(prompt_batch, llm:vllm.entrypoints.llm.LLM):
    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1000)
    # Get api response
    response = do_gemma_inference(prompt, engineering_id, llm, sampling_params)
    pass
    
def gemma_predictions(prompt_batch:list[dict], llm:vllm.entrypoints.llm.LLM):
    """ Function for zero and few-shot predictions, since they have similarly structured output.
    """
    # Define sampling parameters. We need fewer if CoT is not needed. This speed up the model, if it generates non-sense
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    
    for prompt_dict in prompt_batch:
        # Get prompt information
        prompt = prompt_dict.get("prompt")
        paragraph_id = prompt_dict.get("paragraph_id")
        prompt_type = prompt_dict.get("prompt_type")
        # Get gemma prediction
        prediction = do_gemma_inference(prompt, paragraph_id, llm, sampling_params)

        insert_prediction(paragraph_id, 'gemma-3-27b-it', prompt_type, 'zero_shot', prediction, None, None)
    

def process_test_set(llm:vllm.entrypoints.llm.LLM):
    to_be_predicted_batch = get_engineering_data(sample_size=9999)
    prompt_types = get_prompt_list(cot=False, few_shot=False)
    
    for prompt_type in prompt_types:
        print(f"Calling api with samples and prompt: {prompt_type}...")
        prompt_batch = get_test_batch(to_be_predicted_batch, prompt_type)
        gemma_predictions(prompt_batch, llm)
           
        

def main():
    """
    Main function to load the LLM and start the predictions.
    """
    # Set token as an environment variable for vLLM
    print("Retrieving Huggingface API key...")
    hf_token = get_hf_api_key()
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    else:
        print("Hugging Face API key not found. Exiting.")
        return

    # vLLM Initialization
    start_time = time.time()
    model_name = "google/gemma-3-27b-it"
    
    print(f"Initializing LLM '{model_name}' with vLLM...")
    print("This may take several minutes...")
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=2,  # Use 2 GPUs
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return

    elapsed_time = (time.time() - start_time) / 60
    print(f"--- LLM setup complete in {elapsed_time:.2f} min. Ready for prompts. ---")
    process_test_set(llm)



if __name__ == "__main__":
    main()
