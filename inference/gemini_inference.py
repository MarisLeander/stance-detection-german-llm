import json
import time
from tqdm import tqdm
from google import genai
from pathlib import Path
from google.genai import types
from google.genai.errors import ClientError
from google.genai.errors import ServerError
from inference_helper import get_formatted_prompt, get_prompt_list, get_engineering_data, insert_prediction, get_test_batch

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

def do_gemini_api_call(prompt:str, input_data, client:genai.Client) -> dict:
    """ Calls the Gemini API with the given prompt and input data

    Args:
        prompt: The prompt to be sent to the API
        input_data: The data to be sent to the API
        client: The Google Gemini client
    Returns:
        The response from the API
    """
    response = None
    successful_api_call = False
    i = 0

    while not successful_api_call:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True
                    )
                )
            )
            successful_api_call = True
        except (ClientError, ServerError) as e:
            i += 1
            if i == 5:
                error = f"""Failed to call the gemini api 5 times
                        Error: {e}
                        Input data: {input_data}"""
                write_log(error, "api_call_error.txt")
                return {}  # Return an empty dict to avoid None
            else:
                time.sleep(5)  # Sleep before retrying
                continue

    # If we exit the loop normally, we got a successful API call
    return response


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

def gemini_predictions(prompt_batch:list[dict], client:genai.Client):
    """ Function for zero and few-shot predictions, since they have similarly structured output.
    """
    
    for prompt_dict in tqdm(prompt_batch, desc="Processing paragraphs..."):
        # Get prompt information
        prompt = prompt_dict.get("prompt")
        paragraph_id = prompt_dict.get("paragraph_id")
        prompt_type = prompt_dict.get("prompt_type")
        # Get api response
        response = do_gemini_api_call(prompt, paragraph_id, client)
        thoughts, prediction = parse_gemini_response(response)
        # Insert pred into db
        insert_prediction(paragraph_id, 'gemini-2.5-pro', prompt_type, 'zero_shot', prediction, thoughts, None)
    
           
def process_test_set():
    client = get_client()
    to_be_predicted_batch = get_engineering_data(sample_size=1)
    prompt_types = get_prompt_list(cot=False, few_shot=False)
    for prompt_type in prompt_types:
        print(f"Calling api with samples and prompt: {prompt_type}...")
        prompt_batch = get_test_batch(to_be_predicted_batch, prompt_type)
        gemini_predictions(prompt_batch, client)
    

def main():
    process_test_set()

if __name__ == "__main__":
    main()














