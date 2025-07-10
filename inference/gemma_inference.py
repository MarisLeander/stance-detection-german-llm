import os
import time
import json
from tqdm import tqdm
from pathlib import Path
import duckdb as db
from duckdb import ConstraintException
import vllm
from vllm import LLM, SamplingParams
import logging
import contextlib
import io

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

def get_formatted_prompt(paragraph:str, group:str, prompt_type:str) -> str:
    prompt = None
    if prompt_type == 'english_vanilla':
        # Vanilla prompt as in \citet{zhang_sentiment_2023}
        prompt = f""" Please perform Stance Detection task. Given the Paragraph of a speech, assign a sentiment label expressed by the speaker towards "{group}" from [’against’, ’favor’, ’none’]. Return label only without any other text.

        Paragraph: {paragraph}
        Label:
        """
    elif prompt_type == 'german_vanilla':
        prompt = f"""Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.
            
            Textabschnitt: {paragraph}
            Label:
        """
    elif prompt_type == 'german_vanilla_expert_v2':
        prompt = f"""Du bist ein Experte für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textausschnitt.
        Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.
            
            Textabschnitt: {paragraph}
            Label:
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
    elif prompt_type == 'german_more_context_cot_v1':
        # CoT prompt
        prompt = f"""
        **Rolle und Ziel:**
        Du agierst als streng unparteiischer Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer spezifisch markierten Gruppe zu klassifizieren.
        
         **Definition der Kategorien:**
            *   **`favour`**: Wähle diese Kategorie nur, wenn der Sprecher sich **eindeutig und direkt** positiv, unterstützend oder wohlwollend gegenüber der Gruppe äußert. Die positive Haltung muss unmissverständlich im Text formuliert sein.
            *   **`against`**: Wähle diese Kategorie nur, wenn der Sprecher sich **eindeutig und direkt** negativ, kritisch oder ablehnend gegenüber der Gruppe äußert. Die negative Haltung muss unmissverständlich im Text formuliert sein.
            *   **`neither`**: Dies ist die **Standardkategorie**. Wähle sie, wenn die Gruppe nur neutral erwähnt wird ODER wenn die Haltung des Sprechers ambivalent, unklar oder nicht eindeutig aus dem Text bestimmbar ist. Im Zweifelsfall, wenn die Kriterien für `favour` oder `against` nicht **zweifelsfrei** erfüllt sind, wähle **immer** `neither`.
        
        **Anweisungen für die Ausgabe:**
        Deine Antwort muss exakt der folgenden Struktur folgen, um eine einfache maschinelle Verarbeitung zu ermöglichen.
        1.  Beginne mit `Gedankengang:`. Beschreibe hier deine Schritt-für-Schritt-Analyse des Textes.
        2.  Beende deine Antwort mit `Finale Haltung:` in einer neuen Zeile, gefolgt von genau einem der drei Schlüsselwörter.
        
        **Beispiel für die geforderte Ausgabe-Struktur:**
        Gedankengang: Der Sprecher kritisiert die Gruppe X, indem er die Worte "Problem" und "inakzeptabel" verwendet. Dies ist eine klar negative Äußerung. Daher ist die Haltung "against".
        Finale Haltung: against
        
        ---
        
        **Aufgabe:**
        
        **Textabschnitt:**
        {paragraph}
        
        **Gruppe:**
        {group}
        
        **Haltung:**
        """
    elif prompt_type == 'german_more_context_cot_v2':
        # CoT prompt
        prompt = f"""
            Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textausschnitt.
            
            **Zentrale Anweisungen für die Analyse:**
            1.  **Nur die Haltung des Sprechers:** Beurteile ausschließlich die Haltung des **Sprechers**, nicht die von anderen Personen, die im Text zitiert oder erwähnt werden.
            2.  **Nur der Text zählt:** Deine Entscheidung muss sich **allein auf den vorgelegten Text** stützen. Verwende kein externes Wissen über den Sprecher, seine politische Partei oder den allgemeinen Kontext.
            3.  **Berichten ist nicht Werten:** Wenn der Sprecher lediglich eine Meinung, eine Handlung oder eine Situation der Gruppe berichtet (z.B. "Die Leistungen der Gruppe sind gesunken"), ohne eine eigene klare positive oder negative Wertung hinzuzufügen, ist die Haltung `neither`. Eine sachliche Feststellung ist keine Haltung.
            
            **Definition der Kategorien:**
            *   **`favour`**: Die Haltung ist **eindeutig und direkt** positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
            *   **`against`**: Die Haltung ist **eindeutig und direkt** negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
            *   **`neither`**: Dies ist die **Standardkategorie im Zweifelsfall**. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt (siehe Zentrale Anweisungen).

            **Anweisungen für die Ausgabe:**
            Deine Antwort muss exakt der folgenden Struktur folgen, um eine einfache maschinelle Verarbeitung zu ermöglichen.
            1.  Beginne mit `Gedankengang:`. Beschreibe hier deine Schritt-für-Schritt-Analyse des Textes.
            2.  Beende deine Antwort mit `Finale Haltung:` in einer neuen Zeile, gefolgt von genau einem der drei Schlüsselwörter.
            
            **Beispiel für die geforderte Ausgabe-Struktur:**
            Gedankengang: Der Sprecher kritisiert die Gruppe X, indem er die Worte "Problem" und "inakzeptabel" verwendet. Dies ist eine klar negative Äußerung. Daher ist die Haltung "against".
            Finale Haltung: against
            
            ---
            
            **Aufgabe:**
            
            **Textabschnitt:**
            {paragraph}
            
            **Gruppe:**
            {group}
            
            **Haltung:**
        """
    return prompt


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
    
def do_gemma_inference(prompt:str, input_data, con:db.DuckDBPyConnection, llm:vllm.entrypoints.llm.LLM, sampling_params) -> str:
    print(type(sampling_params))
    # Generate the response
    responses = run_silently(llm.generate, [prompt], sampling_params)
    # Print the output
    if len(responses) > 1:
        print("Output was longer than 1!!")
    else:
        generated_text = responses[0].outputs[0].text.strip() 
        return generated_text

def gemma_predictions(paragraph:str, group:str, engineering_id:int, prompt_type:str, con:db.DuckDBPyConnection, llm:vllm.entrypoints.llm.LLM, cot:bool=False):
    # Get prompt
    prompt = get_formatted_prompt(paragraph, group, prompt_type)
    if cot:
        # Define sampling parameters
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1000)
        # Get api response
        response = do_gemma_inference(prompt, engineering_id, con, llm, sampling_params)
        # If the response was from an CoT prompt we need to extract the data differently @todo
        pass
    else:
        # Define sampling parameters. We need fewer if CoT is not needed. This speed up the model, if it generates non-sense
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
        # Get api response
        response = do_gemma_inference(prompt, engineering_id, con, llm, sampling_params)
        sql = """
            INSERT INTO predictions (id, model, run, technique, prediction) 
            VALUES (?, ?, ?, ?, ?)
            """
        # Insert prediction into db
        try:
            con.execute(sql, (engineering_id, "gemma-3-27b-it", prompt_type, "zero-shot", response))
        except ConstraintException:
            # If the response is not in 'favour' / 'against' / 'neither' we will input None, signaling it couldnt classify it. 
            # We do this, to later report our failure score
            con.execute(sql, (engineering_id, "gemma-3-27b-it", prompt_type, "zero-shot", None))
    

def process_test_set(con:db.DuckDBPyConnection, llm:vllm.entrypoints.llm.LLM):
    test_set = con.execute("SELECT * FROM engineering_data JOIN annotated_paragraphs USING(id)").fetchdf()
    prompt_types = ['english_vanilla', 'german_vanilla', 'german_vanilla_expert', 'german_vanilla_expert_v2', 'german_more_context']
    for prompt_type in prompt_types:
    
        for _,row in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Calling api with samples and prompt: {prompt_type}"):
            paragraph = row['inference_paragraph']
            engineering_id = row['id']
            group = row['group_text']

            gemma_predictions(paragraph, group, engineering_id, prompt_type, con, llm)
           
        

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
    # Get db connection
    con = connect_to_db()
    process_test_set(con, llm)
    con.close()



if __name__ == "__main__":
    main()
