import time
from tqdm import tqdm
from datetime import datetime
import duckdb as db
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.genai.errors import ServerError
from pathlib import Path
from duckdb import ConstraintException
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

def get_formatted_prompt(paragraph:str, group:str, prompt_type:str) -> str:
    prompt = None
    if prompt_type == "thinking_guideline":
        # Zero shot
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
    elif prompt_type == "thinking_guideline_higher_standards":
        # Zero shot with higher standards for favour / against so the model rather uses neither
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
    elif prompt_type == 'german_vanilla_expert':
        prompt = f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textausschnitt.
        Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.
            
            Textabschnitt: {paragraph}
            Label:
        """
    elif prompt_type == 'german_more_context':
        # Simple zero shot prompt for gemma3 which doesn't incorporate CoT
        prompt = f"""
            **Rolle und Ziel:**
            Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textausschnitt.
            
            **Zentrale Anweisungen für die Analyse:**
            1.  **Nur die Haltung des Sprechers:** Beurteile ausschließlich die Haltung des **Sprechers**, nicht die von anderen Personen, die im Text zitiert oder erwähnt werden.
            2.  **Nur der Text zählt:** Deine Entscheidung muss sich **allein auf den vorgelegten Text** stützen. Verwende kein externes Wissen über den Sprecher, seine politische Partei oder den allgemeinen Kontext.
            3.  **Berichten ist nicht Werten:** Wenn der Sprecher lediglich eine Meinung, eine Handlung oder eine Situation der Gruppe berichtet (z.B. "Die Leistungen der Gruppe sind gesunken"), ohne eine eigene klare positive oder negative Wertung hinzuzufügen, ist die Haltung `neither`. Eine sachliche Feststellung ist keine Haltung.
            
            **Definition der Kategorien:**
            *   **`favour`**: Die Haltung ist **eindeutig und direkt** positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
            *   **`against`**: Die Haltung ist **eindeutig und direkt** negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
            *   **`neither`**: Dies ist die **Standardkategorie im Zweifelsfall**. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt (siehe Zentrale Anweisungen).
            
            **Anforderung an die Antwort:**
            Deine Antwort muss **ausschließlich** eines der drei folgenden Wörter enthalten. Gib keine Erklärungen, Begrüßungen oder Satzzeichen aus.
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
    elif prompt_type == 'english_vanilla':
        # Vanilla prompt as in \citet{zhang_sentiment_2023}
        prompt = f""" Please perform Stance Detection task. Given the Paragraph of a speech, assign a sentiment label expressed by the speaker towards "{group}" from [’against’, ’favor’, ’none’]. Return label only without any other text.

        Paragraph: {paragraph}
        Label:
        """
    return prompt
    
def gemini_predictions(paragraph:str, group:str, engineering_id:int, prompt_type:str, con:db.DuckDBPyConnection):
    client = get_client()
    # Get prompt
    prompt = get_formatted_prompt(paragraph, group, prompt_type)
    # Get api response
    response = do_gemini_api_call(prompt, engineering_id, client)
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
    # Insert prediction into db
    try:
        con.execute(sql, (engineering_id, "gemini-2.5-pro", prompt_type,"zero-shot", response, thoughts))
    except ConstraintException:
        # If the model fails to provide output out of ['favour', 'against', 'neither']
        con.execute(sql, (engineering_id, "gemini-2.5-pro", prompt_type,"zero-shot", None, thoughts))
    

def process_test_set(con:db.DuckDBPyConnection):
    test_set = con.execute("SELECT * FROM engineering_data JOIN annotated_paragraphs USING(id) LIMIT 20").fetchdf()
    prompt_types = ["thinking_guideline", "thinking_guideline_higher_standards", "german_vanilla_expert", "german_more_context", "english_vanilla"]
    for prompt_type in prompt_types:
        for _,row in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Calling api with samples and prompt: {prompt_type}"):
            paragraph = row['inference_paragraph']
            engineering_id = row['id']
            group = row['group_text']
            agreed_label = row['agreed_label']

            gemini_predictions(paragraph, group, engineering_id, prompt_type, con)
           
        
    

def main():
    con = connect_to_db()
    process_test_set(con)

if __name__ == "__main__":
    main()














