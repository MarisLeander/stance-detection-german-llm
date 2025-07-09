import time
from tqdm import tqdm
from datetime import datetime
import duckdb as db
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.genai.errors import ServerError
from pathlib import Path
import json

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

def connect_to_db() -> db.DuckDBPyConnection:
    """
    Connect to a DuckDB database.

    Args:
        db_path (str): The path to the DuckDB database file.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the DuckDB database.
    """
    home_dir = Path.home()
    db_path = home_dir / "stance-detection-german-llm" / "data" / "database" / "german-parliament.duckdb"
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

def do_api_call(prompt:str, input_data, client:genai.Client) -> dict:
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
                time.sleep(30)  # Sleep before retrying
                continue

    # If we exit the loop normally, we got a successful API call
    return response

def call_gemini_api(paragraph:str, group:str, engineering_id:int, client:genai.Client, prompt_num:int) -> str:
    prompt = None
    if prompt_num == 1:
        prompt = f"""
            **Rolle und Ziel:**
            Du bist ein unparteiischer Experte für politische Diskursanalyse. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer spezifischen sozialen Gruppe zu identifizieren.
            
            **Kontext der Aufgabe:**
            Du erhältst einen Textabschnitt aus einer Rede, die im Deutschen Bundestag gehalten wurde. In diesem Text ist die Nennung einer sozialen Gruppe mit `<span>`-Tags markiert. Deine Analyse muss sich ausschließlich auf die Haltung des Sprechers gegenüber dieser markierten Gruppe beziehen, wie sie in diesem spezifischen Abschnitt zum Ausdruck kommt.
            
            **Anweisungen:**
            1.  Lies den bereitgestellten **Textabschnitt** sorgfältig.
            2.  Identifiziere die markierte **Gruppe**.
            3.  Analysiere die Wortwahl, den Ton und den argumentativen Kontext des Sprechers in Bezug auf diese Gruppe.
            4.  Klassifiziere die Haltung des Sprechers in eine der drei folgenden Kategorien.
            
            **Definition der Kategorien:**
            *   **`favour`**: Wähle diese Kategorie, wenn der Sprecher sich positiv, unterstützend, verteidigend oder wohlwollend gegenüber der Gruppe äußert. Dies ist der Fall, wenn der Sprecher die Gruppe lobt, ihre Anliegen als legitim darstellt oder Maßnahmen zu ihrem Vorteil fordert.
            *   **`against`**: Wähle diese Kategorie, wenn der Sprecher sich negativ, kritisch, abwertend oder ablehnend gegenüber der Gruppe äußert. Dies ist der Fall, wenn der Sprecher die Gruppe oder ihre Handlungen kritisiert, vor ihr warnt oder sie für Probleme verantwortlich macht.
            *   **`neither`**: Wähle diese Kategorie, wenn der Sprecher die Gruppe lediglich erwähnt, ohne eine klare positive oder negative Wertung vorzunehmen. Dies umfasst rein informative, statistische oder neutrale Nennungen, oder wenn die Haltung aus dem Abschnitt nicht eindeutig bestimmbar ist.
            
            **Anforderung an die Antwort:**
            Deine Antwort muss **ausschließlich** eines der drei folgenden Wörter enthalten, ohne zusätzliche Erklärungen, Begrüßungen oder Satzzeichen:
            `favour`
            `against`
            `neither`
            
            ---
            
            **Aufgabe:**
            
            **Textabschnitt:**
            {paragraph}
            
            **Gruppe:**
            {group}
            
            **Haltung:**
            """
    elif prompt_num == 2:
        prompt = f"""
            **Rolle und Ziel:**
        Du agierst als streng unparteiischer Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer spezifisch markierten Gruppe zu klassifizieren.
        
        **Grundprinzip der Analyse:**
        Deine Analyse muss sich **ausschließlich** auf die expliziten Aussagen im vorgelegten Textabschnitt stützen. Die Haltung muss **direkt aus dem Wortlaut** ableitbar sein. Interpretiere oder schlussfolgere nicht über den Text hinaus. Insbesondere bei Ambiguität oder Unklarheit ist höchste Vorsicht geboten.
        
        **Anweisungen:**
        1.  Lies den bereitgestellten **Textabschnitt** und identifiziere die markierte **Gruppe**.
        2.  Analysiere die Haltung des Sprechers **ausschließlich** basierend auf den Worten, die er in diesem Abschnitt verwendet.
        3.  Wähle eine der folgenden drei Kategorien.
        
        **Definition der Kategorien:**
        *   **`favour`**: Wähle diese Kategorie nur, wenn der Sprecher sich **eindeutig und direkt** positiv, unterstützend oder wohlwollend gegenüber der Gruppe äußert. Die positive Haltung muss unmissverständlich im Text formuliert sein.
        *   **`against`**: Wähle diese Kategorie nur, wenn der Sprecher sich **eindeutig und direkt** negativ, kritisch oder ablehnend gegenüber der Gruppe äußert. Die negative Haltung muss unmissverständlich im Text formuliert sein.
        *   **`neither`**: Dies ist die **Standardkategorie**. Wähle sie, wenn die Gruppe nur neutral erwähnt wird ODER wenn die Haltung des Sprechers ambivalent, unklar oder nicht eindeutig aus dem Text bestimmbar ist. Im Zweifelsfall, wenn die Kriterien für `favour` oder `against` nicht **zweifelsfrei** erfüllt sind, wähle **immer** `neither`.
        
        **Anforderung an die Antwort:**
        Deine Antwort muss **ausschließlich** eines der drei folgenden Wörter enthalten, ohne zusätzliche Erklärungen, Begrüßungen oder Satzzeichen:
        `favour`
        `against`
        `neither`
        
        ---
        
        **Aufgabe:**
        
        **Textabschnitt:**
        {paragraph}
        
        **Gruppe:**
        {group}
        
        **Haltung:**
        """
    input_data = ("engineering_id", engineering_id)
    return do_api_call(prompt, input_data, client)
    

def process_test_set(con:db.DuckDBPyConnection, client:genai.Client):
    test_set = con.execute("SELECT * FROM engineering_data JOIN annotated_paragraphs USING(id) WHERE id NOT IN (SELECT id FROM predictions) ORDER BY RANDOM() LIMIT 20").fetchdf()
    for i in range(1,3):
        prompt_template = f"prompt_0{i}"
    
        for _,row in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Calling api with samples and prompt_0{i} out of 02"):
            paragraph = row['inference_paragraph']
            engineering_id = row['id']
            group = row['group_text']
            agreed_label = row['agreed_label']
            # Get api response
            response = call_gemini_api(paragraph, group, engineering_id, client, i)
            # Analyse response
            first_candidate = response.candidates[0]
            thoughts = ""
            response = ""
            for part in first_candidate.content.parts:
                if part.thought:
                    thoughts += part.text
                else:
                    response += part.text
    
            sql = """
                INSERT INTO predictions (id, model, run, technique, prediction, thoughts) 
                VALUES (?, ?, ?, ?, ?, ?)
                """
            con.execute(sql, (engineering_id, "gemini-2.5-pro", prompt_template,"zero-shot", response, thoughts))
        
    

def main():
    client = get_client()
    con = connect_to_db()
    process_test_set(con, client)

if __name__ == "__main__":
    main()














