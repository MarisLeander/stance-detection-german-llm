import os
import csv
import json
import time
import argparse
import inference_helper as ih

from tqdm import tqdm
from google import genai
from pathlib import Path
from datetime import datetime
from google.genai import types
from google.genai.errors import ClientError
from google.genai.errors import ServerError

def get_gemini_api_key() -> str:
    """Gets the users Google Gemini api key from the config file

    Args:
        None

    Returns:
        The Google Gemini api key of the user
    """

    home_dir = Path.home()
    path = home_dir / "stance-detection-german-llm" / "secrets.json"
    try:
        with open(path, "r") as config_file:
            config = json.load(config_file)
        return config.get("gemini_api_key")
    except FileNotFoundError:
        print(f"Error: secrets file not found at {path}")
        return None


def get_client() -> genai.Client:
    return genai.Client(api_key=get_gemini_api_key())

def write_log(msg: str, logfile: str):
    """Writes a message to the log file.

    Args:
        msg: The message to write to the log file
        logfile: The name of the log file

    Returns:
        None
    """
    home = Path.home()
    
    file_path = home / "stance-detection-german-llm" / "inference" / "gemini_inference_logs" / logfile
    with open(file_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp}\n{msg}\n\n")

def do_gemini_api_call(instruction_string:str, user_prompt:str, input_data, client:genai.Client) -> dict:
    """ Calls the Gemini API with the given prompt and input data

    Args:
        prompt: The prompt to be sent to the API
        input_data: The data to be sent to the API
        client: The Google Gemini client
    Returns:
        The response from the API
    """
    client_config = None
    if instruction_string:
        client_config = types.GenerateContentConfig(system_instruction=instruction_string, temperature=0.0, thinking_config=types.ThinkingConfig(include_thoughts=True))
    else:
        client_config = types.GenerateContentConfig(temperature=0.0, thinking_config=types.ThinkingConfig(include_thoughts=True))
    
    response = None
    successful_api_call = False
    i = 0

    while not successful_api_call:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=user_prompt,
                config=client_config
            )
            successful_api_call = True
        except (ClientError, ServerError) as e:
            i += 1
            if i == 2:
                error = f"""Failed to call the gemini api 5 times
                        Error: {e}
                        Input data: {input_data}"""
                write_log(error, "api_call_error.txt")
                return None
            else:
                time.sleep(5)  # Sleep before retrying
                continue

    # If we exit the loop normally, we got a successful API call
    return response

def append_to_csv(result_data: dict, filepath: str):
    """
    Appends a single dictionary of data to a CSV file in a thread-safe manner.
    Writes the header only if the file is new or empty.

    Args:
        result_data (dict): A dictionary containing the prediction results.
        filepath (str): The path to the output CSV file.
    """

    # Check if the file already exists to determine if we need a header
    file_exists = os.path.isfile(filepath)
    
    # Open the file in append mode ('a')
    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        # Define the column names from the keys of your data dictionary
        fieldnames = result_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is new or empty, write the header row first
        if not file_exists or os.path.getsize(filepath) == 0:
            writer.writeheader()
        
        # Write the actual data row
        writer.writerow(result_data)

def parse_gemini_response(response:dict) -> tuple[str,str]:
    first_candidate = response.candidates[0]
    thoughts = ""
    prediction = ""
    for part in first_candidate.content.parts:
        if part.thought:
            thoughts += part.text
        else:
            prediction += part.text
    return thoughts, prediction

def gemini_predictions(prompt_batch:list[dict], client:genai.Client, technique:str, it_setup:bool=False, engineering:bool=False):
    """ Function for zero and few-shot predictions, since they have similarly structured output.
    """
    
    for prompt_dict in tqdm(prompt_batch, desc="Processing paragraphs..."):
        # Check if we already handled the exact config and prompt
        model = 'gemini-2.5-pro'
        paragraph_id = prompt_dict.get("paragraph_id")
        prompt_type = prompt_dict.get("prompt_type")
        
        processed = ih.already_processed(paragraph_id, model, prompt_type, technique)
        if processed:
            # We dont need to waste credits then...
            continue

        system_prompt = None
        user_prompt = None
        if it_setup:
            # If we use the split prompts, we need to adjust our user and system prompt.
            system_prompt = prompt_dict.get("message")[0].get("content")
            user_prompt =  prompt_dict.get("message")[1].get("content")
        else:
            # If we don't do split prompts our user prompt is the complete prompt.
            user_prompt = prompt_dict.get("prompt")
            
        # Get api response
        response = do_gemini_api_call(system_prompt, user_prompt, paragraph_id, client)
        thoughts, prediction = parse_gemini_response(response)
        thinking_process = None
        if technique == "CoT":
            thinking_process = prediction
            prediction = ih.extract_stance_cot(thinking_process)

        if engineering:
            ih.insert_prediction(engineering, paragraph_id, model, prompt_type, technique, prediction, thoughts, thinking_process)
        elif it_setup:
            result_data = {
                "engineering": engineering,
                "paragraph_id": paragraph_id,
                "model": model,
                "prompt_type": prompt_type,
                "technique": technique,
                "prediction": prediction,
                "thoughts": thoughts,
                "thinking_process": thinking_process,
                "prompt": (system_prompt + "\n" + user_prompt)
            }
            append_to_csv(result_data, (prompt_type + '_' + technique + '.csv'))
        else:
            result_data = {
                "engineering": engineering,
                "paragraph_id": paragraph_id,
                "model": model,
                "prompt_type": prompt_type,
                "technique": technique,
                "prediction": prediction,
                "thoughts": thoughts,
                "thinking_process": thinking_process,
                "prompt": user_prompt
            }
            append_to_csv(result_data, (prompt_type + '_' + technique + '.csv'))
            
        
           
def process_engineering_set():
    client = get_client()
    to_be_predicted_batch = ih.get_engineering_data(sample_size=9999)

    # #  ****** Process it-split-prompts ******
    # 1. Process zero-shot split-prompts
    prompt_types = ih.get_engineering_prompt_list(cot=False, few_shot=False, it_setup=True)
    for prompt_type in prompt_types:
        print(f"Processing {prompt_type}")
        # Get batch of to-be-processed prompts
        prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, engineering=True)
        gemini_predictions(prompt_batch, client, technique='zero-shot', it_setup=True, engineering=True)
        
    # 2. Process cot split-prompts
    prompt_types = ih.get_engineering_prompt_list(cot=True, few_shot=False, it_setup=True)
    for prompt_type in prompt_types:
        print(f"Processing {prompt_type}")
         # Get batch of to-be-processed prompts
        prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, engineering=True)
        gemini_predictions(prompt_batch, client, technique='CoT', it_setup=True, engineering=True)

    # 3. Process few-shot split-prompts
    prompt_types = ih.get_engineering_prompt_list(cot=False, few_shot=True, it_setup=True)
    for prompt_type in prompt_types:
        # Get batch of to-be-processed prompts
        print(f"Processing 1-shot for {prompt_type}")
        prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='1-shot', engineering=True)
        gemini_predictions(prompt_batch, client, technique='1-shot', it_setup=True, engineering=True)
        print(f"Processing 5-shot for {prompt_type}")
        prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='5-shot', engineering=True)
        gemini_predictions(prompt_batch, client, technique='5-shot', it_setup=True, engineering=True)
        print(f"Processing 10-shot for {prompt_type}")
        prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='10-shot', engineering=True)
        gemini_predictions(prompt_batch, client, technique='10-shot', it_setup=True, engineering=True)

    #  ****** Process non-it-split-prompts ******
    # Process zero-shot prompts
    prompt_types = ih.get_engineering_prompt_list(cot=False, few_shot=False)
    
    for prompt_type in prompt_types:
        print(f"Calling api with samples and prompt: {prompt_type}...")
        prompt_batch = ih.get_test_batch(to_be_predicted_batch, prompt_type, engineering=True)
        gemini_predictions(prompt_batch, client, technique='zero_shot', engineering=True)
        

    # Process few-shot prompts
    prompt_types_fs = ih.get_engineering_prompt_list(cot=False, few_shot=True)
    
    for prompt_type in prompt_types_fs:
        print(f"Calling gemini api with samples and few-shot prompt: {prompt_type}...")
        print("Processing 1-shot prompts...")
        prompt_batch_fs1 = ih.get_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='1-shot', engineering=True)
        gemini_predictions(prompt_batch_fs1, client, technique='1-shot', engineering=True)
        print("Processing 5-shot prompts...")
        prompt_batch_fs5 = ih.get_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='5-shot', engineering=True)
        gemini_predictions(prompt_batch_fs5, client, technique='5-shot', engineering=True)
        print("Processing 10-shot prompts...")
        prompt_batch_fs10 = ih.get_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots='10-shot', engineering=True)
        gemini_predictions(prompt_batch_fs10, client, technique='10-shot', engineering=True)

    # Process CoT prompts
    prompt_types_cot = ih.get_engineering_prompt_list(cot=True, few_shot=False)

    for prompt_type in prompt_types_cot:
        print(f"Calling gemma model with samples and CoT prompt: {prompt_type}...")
        prompt_batch_cot = ih.get_test_batch(to_be_predicted_batch, prompt_type, cot=True, engineering=True)
        gemini_predictions(prompt_batch_cot, client, technique='CoT', engineering=True)



def process_test_set(api_key:str, prompt_type:str, technique:str, shots:str=None):
    client = genai.Client(api_key=api_key)
    to_be_predicted_batch = ih.get_test_data()
    
    it_setup = False

    if prompt_type.startswith("it-"):
        it_setup = True    
        
    prompt_batch = None
    
    if it_setup:    
        if shots:
            print("Preparing few-shot examples....")
            prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, few_shot=True, shots=shots, engineering=False)
        else:
            prompt_batch =  ih.get_split_test_batch(to_be_predicted_batch, prompt_type, engineering=False)
    elif not it_setup and technique == "CoT":
        prompt_batch = ih.get_test_batch(to_be_predicted_batch, prompt_type, cot=True, engineering=True)
    
    (f"Processing {prompt_type}")
    gemini_predictions(prompt_batch, client, technique, it_setup=it_setup, engineering=False)
        
    

def main():
    #process_engineering_set()
    # 1. Set up the argument parser with a description
    parser = argparse.ArgumentParser(
        description="Run Gemini inference for a specific prompt configuration.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    # 2. Add arguments
    parser.add_argument(
        "--api-key", 
        type=str, 
        required=True, 
        help="Your Google Gemini API key."
    )
    parser.add_argument(
        "--prompt-type", 
        type=str, 
        required=True, 
        help="The specific prompt template to use.\n(e.g., 'it-thinking_guideline_higher_standards')"
    )
    parser.add_argument(
        "--technique", 
        type=str, 
        required=True, 
        help="The prompting technique to record in the database.\n(e.g., 'zero-shot', 'CoT', '1-shot')"
    )
    parser.add_argument(
        "--shots", 
        type=str, 
        default=None, 
        help="The number of shots for few-shot prompting.\n(e.g., '1-shot', '5-shot').\nDefaults to None if not provided."
    )
    
    # 3. Parse the arguments from the command line
    args = parser.parse_args()
    
    # 4. Call the processing function with the provided arguments
    process_test_set(
        api_key=args.api_key,
        prompt_type=args.prompt_type,
        technique=args.technique,
        shots=args.shots
    )

if __name__ == "__main__":
    main()














