import duckdb as db
import pandas as pd

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

def get_prompt_list(cot:bool=False, few_shot:bool=False) -> list[str]:
    """ This function returns a list of prompts for the model to benchmark on.

    Args:
        cot (bool): Indicates wheter the model needs CoT prompts or not
        few_shot (bool): Indicates wheter the model needs few_shot prompts or not
    """
    if cot:
        pass #@todo
    elif few_shot:
        pass #@todo
    else:
        return ["thinking_guideline", "thinking_guideline_higher_standards", "german_vanilla", "german_vanilla_expert", "german_more_context", "english_vanilla"]
    
def get_formatted_prompt(paragraph:str, group:str, prompt_type:str) -> str:
    if prompt_type == "thinking_guideline":
        # Zero shot
        return f"""
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
        return f"""
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
    elif prompt_type == 'german_vanilla':
        return f"""Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.
            
            Textabschnitt: {paragraph}
            Label:
        """
    elif prompt_type == 'german_vanilla_expert':
        return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textausschnitt.
        Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.
            
            Textabschnitt: {paragraph}
            Label:
        """
    elif prompt_type == 'german_more_context':
        # Simple zero shot prompt for gemma3 which doesn't incorporate CoT
        return f"""
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
        return f""" Please perform Stance Detection task. Given the Paragraph of a speech, assign a sentiment label expressed by the speaker towards "{group}" from [’against’, ’favor’, ’neither’]. Return label only without any other text.

        Paragraph: {paragraph}
        Label:
        """

def get_test_batch(batch:pd.DataFrame, prompt_type:str)
def get_test_data(sample_size:int=1)
    sql = """
        SELECT id, inference_paragraph, group_text 
        FROM engineering_data 
            JOIN annotated_paragraphs USING(id) 
        ORDER BY RANDOM() 
        LIMIT ?
    """
    test_set = con.execute(sql, (sample_size,)).fetchdf()
    prompt_types = ["thinking_guideline", "thinking_guideline_higher_standards", "german_vanilla_expert", "german_more_context", "english_vanilla"]
    for prompt_type in prompt_types:
        for _,row in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Calling api with samples and prompt: {prompt_type}"):
            paragraph = row['inference_paragraph']
            engineering_id = row['id']
            group = row['group_text']
            agreed_label = row['agreed_label']

def insert_prediction(id:int, model:str, prompt_type:str, technique:str, prediction:str, thinking_process:str, thoughts:str):
    """ This function is used to insert a models prediction into our db.

    Args:
        id (int): The id of the predicted paragraph (corresponds to an annotated paragraph)
        model (str): The name of our model (e.g. 'gemini-2.5-pro')
        prompt_type (str): Corresponds to a prompt template
        technique (str): Prompting technique (e.g. 'zero-shot')
        prediction (str): The models predicted stance (e.g. 'favour')
        thinking_process (str): The output thinking process of reasoning models (e.g. for deepseek r1 everything in the <think> tag)
        thoughts (str): The thoughts of the CoT process
    """
    con = connect_to_db()
    con.begin()
    sql = """
        INSERT INTO predictions (id, model, prompt_type, technique, prediction, thinking_process, thoughts) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING --If a certain technique was already benchmarked, do nothing.
        """
    # Insert prediction into db
    try:
        con.execute(sql, (id, prompt_type, technique, prediction, thinking_process, thoughts))
        con.commit()
        con.close()
    except ConstraintException:
        # If the model fails to provide output out of ['favour', 'against', 'neither']
        con.execute(sql, (id, prompt_type, technique, None, thinking_process, thoughts))
        con.commit()
        con.close()
