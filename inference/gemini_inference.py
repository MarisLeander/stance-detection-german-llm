import time
from datetime import datetime

import duckdb as db
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.genai.errors import ServerError
from pathlib import Path

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
        with open("path", "r") as config_file:
            config = json.load(config_file)
        return config.get("gemini_api_key")
    except FileNotFoundError:
        print(f"Error: secrets file not found at {path}")
        return None

def connect_to_db(db_path: str) -> db.DuckDBPyConnection:
    """
    Connect to a DuckDB database.

    Args:
        db_path (str): The path to the DuckDB database file.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the DuckDB database.
    """
    return db.connect(database=db_path, read_only=False)

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

def do_api_call(prompt, input_data, client) -> dict:
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
                    temperature=0.0
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
                time.sleep(30)  # Sleep before retrying
                continue

    # If we exit the loop normally, we got a successful API call
    return response
#%%
prompt = "Hello, my name is:"
client = get_client()
print(do_api_call(prompt, "test", client))
#%%
