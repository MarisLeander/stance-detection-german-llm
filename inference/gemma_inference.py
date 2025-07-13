import os
import io
import time
import json
import vllm
import contextlib
from pathlib import Path
from vllm import LLM, SamplingParams
import inference_helper as ih
from inference_helper import get_formatted_prompt, get_prompt_list, get_engineering_data, insert_prediction, insert_batch, get_test_batch, get_split_test_batch

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


def gemma_split_batch_inference(
    prompt_batch: list[dict],
    llm:vllm.entrypoints.llm.LLM,
    technique: str
):
    sampling_params = None
    if technique == "CoT":
        # If the model should produce a chain-of-thought we need more output tokens
        sampling_params = SamplingParams(temperature=0.0, max_tokens=5000)
    else:
        # If not we only need a label, which need significantly less tokens
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    # Get the tokenizer from the llm object
    tokenizer = llm.get_tokenizer()

    # Prepare all formatted prompt strings in a list
    print("Applying chat templates to all prompts...")
    prompts_to_generate = [
        tokenizer.apply_chat_template(
            prompt_dict.get("message"),
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt_dict in prompt_batch
    ]
    # Make a SINGLE call to llm.generate with the entire batch
    print(f"Generating responses for {len(prompts_to_generate)} prompts in one batch...")
    outputs = llm.generate(prompts_to_generate, sampling_params)

    # Process the results and insert into the database
    print("Processing results and inserting into database...")
    records_to_insert = []
    # We zip the original prompt_batch with the outputs to maintain metadata
    for prompt_dict, output in zip(prompt_batch, outputs):
        paragraph_id = prompt_dict.get("paragraph_id")
        prompt_type = prompt_dict.get("prompt_type")
        generated_text = output.outputs[0].text.strip()

        # If technique is chain-of-thought we need to parse our prediction and insert the CoT as thinking_process
        if technique == "CoT":
            records_to_insert.append({
                "id": paragraph_id, 
                "model": 'gemma-3-27b-it', 
                "prompt_type": prompt_type, 
                "technique": technique, 
                "prediction": ih.extract_stance_cot(generated_text), 
                "thinking_process": generated_text,
                "thoughts": None # Placeholder
            })
        else:
            records_to_insert.append({
                "id": paragraph_id, 
                "model": 'gemma-3-27b-it', 
                "prompt_type": prompt_type, 
                "technique": technique, 
                "prediction": generated_text, 
                "thinking_process": None, # Placeholder
                "thoughts": None # Placeholder
            })
            
    print("Inserting predictions into db")
    insert_batch(records_to_insert)  

def gemma_batch_inference(
    prompt_batch: list[dict],
    llm: LLM,
    technique: str
):
    """
    Processes a batch of pre-formatted prompts with vLLM.

    This version assumes each dictionary in 'prompt_batch' contains a 'prompt'
    key with the full, final prompt string, and does NOT apply a chat template.
    """
    sampling_params = None
    if technique == "CoT":
        # If the model should produce a chain-of-thought we need more output tokens
        sampling_params = SamplingParams(temperature=0.0, max_tokens=5000)
    else:
        # If not we only need a label, which need significantly less tokens
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    # Directly extract the prompt string from each dictionary
    print("Extracting pre-formatted prompts from the batch...")
    prompts_to_generate = [
        prompt_dict.get("prompt") for prompt_dict in prompt_batch
    ]

    # Make a SINGLE call to llm.generate with the entire batch
    print(f"Generating responses for {len(prompts_to_generate)} prompts in one batch...")
    outputs = llm.generate(prompts_to_generate, sampling_params)

    # Process the results and prepare them for database insertion
    print("Processing results...")
    records_to_insert = []
    # We zip the original prompt_batch with the outputs to maintain metadata
    for prompt_dict, output in zip(prompt_batch, outputs):
        paragraph_id = prompt_dict.get("paragraph_id")
        prompt_type = prompt_dict.get("prompt_type")
        generated_text = output.outputs[0].text.strip()

        # If technique is chain-of-thought we need to parse our prediction and insert the CoT as thinking_process
        if technique == "CoT":
            records_to_insert.append({
                "id": paragraph_id, 
                "model": 'gemma-3-27b-it', 
                "prompt_type": prompt_type, 
                "technique": technique, 
                "prediction": ih.extract_stance_cot(generated_text), 
                "thinking_process": generated_text, # Placeholder
                "thoughts": None # Placeholder
            })
        else:
            records_to_insert.append({
                "id": paragraph_id, 
                "model": 'gemma-3-27b-it', 
                "prompt_type": prompt_type, 
                "technique": technique, 
                "prediction": generated_text, 
                "thinking_process": None, # Placeholder
                "thoughts": None # Placeholder
            })
    print("Inserting predictions into db")
    insert_batch(records_to_insert)        
    
 

def process_test_set(llm:vllm.entrypoints.llm.LLM):
    to_be_predicted_batch = get_engineering_data(sample_size=99999)

    # # ****** Process it-split-prompts ******
    # # 1. Process zero-shot split-prompts
    # prompt_types = ih.get_prompt_list(cot=False, few_shot=False, it_setup=True)
    # for prompt_type in prompt_types:
    #     print(f"Processing {prompt_type}")
    #     # Get batch of to-be-processed prompts
    #     prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type)
    #     gemma_split_batch_inference(prompt_batch, llm, technique='zero-shot')
        
    # # 2. Process cot split-prompts
    # prompt_types = ih.get_prompt_list(cot=True, few_shot=False, it_setup=True)
    # for prompt_type in prompt_types:
    #     print(f"Processing {prompt_type}")
    #      # Get batch of to-be-processed prompts
    #     prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type)
    #     gemma_split_batch_inference(prompt_batch, llm, technique='CoT')

    # 3. Process few-shot split-prompts
    # prompt_types = ih.get_prompt_list(cot=False, few_shot=True, it_setup=True)
    # for prompt_type in prompt_types:
    #     print(f"Processing 1-shot for {prompt_type}")
    #     # Get batch of to-be-processed prompts
    #     prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='1-shot')
    #     gemma_split_batch_inference(prompt_batch, llm, technique='1-shot')
    #     print(f"Processing 5-shot for {prompt_type}")
    #     # Get batch of to-be-processed prompts
    #     prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='1-shot')
    #     gemma_split_batch_inference(prompt_batch, llm, technique='5-shot')
    #     print(f"Processing 10-shot for {prompt_type}")
    #     # Get batch of to-be-processed prompts
    #     prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='1-shot')
    #     gemma_split_batch_inference(prompt_batch, llm, technique='10-shot')
    
    # #  ****** Process non-it-split-prompts ******
    # # Process zero-shot prompts
    # prompt_types_zs = get_prompt_list(cot=False, few_shot=False)
    
    # for prompt_type in prompt_types_zs:
    #     print(f"Calling gemma models with samples and zero-shot prompt: {prompt_type}...")
    #     prompt_batch_zs =  get_test_batch(to_be_predicted_batch, prompt_type)
    #     gemma_batch_inference(prompt_batch_zs, llm, technique='zero-shot')
        
    # Process few-shot prompts
    prompt_types_fs = get_prompt_list(cot=False, few_shot=True)
    
    for prompt_type in prompt_types_fs:
        print(f"Calling gemma model with samples and few-shot prompt: {prompt_type}...")
        print("Processing 1-shot prompts...")
        prompt_batch_fs1 = get_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='1-shot')
        gemma_batch_inference(prompt_batch_fs1, llm, technique='1-shot')
        print("Processing 5-shot prompts...")
        prompt_batch_fs5 = get_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='5-shot')
        gemma_batch_inference(prompt_batch_fs5, llm, technique='5-shot')
        print("Processing 10-shot prompts...")
        prompt_batch_fs10 = get_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='10-shot')
        gemma_batch_inference(prompt_batch_fs10, llm, technique='10-shot')

    # # Process CoT prompts
    # prompt_types_cot = ih.get_prompt_list(cot=True, few_shot=False)

    # for prompt_type in prompt_types_cot:
    #     print(f"Calling gemma model with samples and CoT prompt: {prompt_type}...")
    #     prompt_batch_cot = get_test_batch(to_be_predicted_batch, prompt_type, cot=True)
    #     gemma_batch_inference(prompt_batch_cot, llm, technique='CoT')
           
        

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
    print(f"--- LLM setup complete in {elapsed_time:.2f} min. ---")

    process_test_set(llm)



if __name__ == "__main__":
    main()
