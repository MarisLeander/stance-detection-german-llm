import re
import duckdb as db
import pandas as pd
from pathlib import Path
from duckdb import ConstraintException

def connect_to_db(read_only:bool=False) -> db.DuckDBPyConnection:
    """
    Connect to a DuckDB database.

    Args:
        db_path (str): The path to the DuckDB database file.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the DuckDB database.
    """
    home_dir = Path.home()
    db_path = home_dir / "stance-detection-german-llm" / "data" / "database" / "german-parliament.duckdb"
    return db.connect(database=db_path, read_only=read_only)

def already_processed(paragraph_id:int, model:str, prompt_type:str, technique:str) -> bool:
    """ Checks if a certain model, prompt_type, technique combination already classified the paragraph.

    Args: 
        paragraph_id (int): The id of the predicted paragraph (corresponds to an annotated paragraph)
        model (str): The name of our model (e.g. 'gemini-2.5-pro')
        prompt_type (str): Corresponds to a prompt template
        technique (str): Prompting technique (e.g. 'zero-shot')
        
    Returns:
        bool: True if the model, prompt_type, technique combination already classified the paragraph.
    """

    # SELECT EXISTS returns a boolean: True if the subquery finds any rows, False otherwise.
    sql = "SELECT EXISTS(SELECT 1 FROM predictions WHERE id = ? AND model = ? AND prompt_type = ? AND technique = ?);"
    
    # Execute the query and fetch the single boolean result
    con = connect_to_db(read_only=True)
    exists = con.execute(sql, (paragraph_id, model, prompt_type, technique)).fetchone()[0]
    con.close()
    return exists
    
def get_engineering_prompt_list(cot:bool=False, few_shot:bool=False, it_setup:bool=False) -> list[str]:
    """ This function returns a list of prompts for the model to benchmark on, for prompt_engineering.

    Args:
        cot (bool): Indicates wheter the model needs CoT prompts or not
        few_shot (bool): Indicates wheter the model needs few_shot prompts or not
    """
    if cot:
        if it_setup:
            return ["it-german_vanilla_expert_more_context_cot", "it-thinking_guideline_higher_standards_cot", "it-thinking_guideline_cot"]
        else:
            return ["german_vanilla_expert_more_context_cot", "thinking_guideline_higher_standards_cot", "thinking_guideline_cot"]
    elif few_shot:
        if it_setup:
            return ["it-german_vanilla", "it-german_vanilla_expert", "it-german_vanilla_expert_more_context", "it-thinking_guideline_higher_standards", "it-thinking_guideline"]
            # return ["it-thinking_guideline_higher_standards"]
        else:
            return ["german_vanilla", "german_vanilla_expert", "german_vanilla_expert_more_context", "thinking_guideline_higher_standards", "thinking_guideline"]
            # return ["thinking_guideline_higher_standards"]
    else:
        if it_setup:
            return ["it-thinking_guideline", "it-thinking_guideline_higher_standards", "it-german_vanilla", "it-german_vanilla_expert", "it-german_vanilla_expert_more_context", "it-german_more_context", "it-english_vanilla"]
        else:
            return ["thinking_guideline", "thinking_guideline_higher_standards", "german_vanilla", "german_vanilla_expert", "german_vanilla_expert_more_context", "german_more_context", "english_vanilla"]

# def get_test_prompt_list(cot:bool=False, few_shot:bool=False, it_setup:bool=False) -> list[str]:
#     """ This function returns a list of prompts for the model to benchmark on our test data.

#     Args:
#         cot (bool): Indicates wheter the model needs CoT prompts or not
#         few_shot (bool): Indicates wheter the model needs few_shot prompts or not
#     """
#     if cot:
#         if it_setup:
#             return ["it-german_vanilla_expert_more_context_cot", "it-thinking_guideline_higher_standards_cot"]
#         else:
#             return ["german_vanilla_expert_more_context_cot", "thinking_guideline_higher_standards_cot"]
#     elif few_shot:
#         if it_setup:
#             return ["it-thinking_guideline_higher_standards"]
#         else:
#             return ["thinking_guideline_higher_standards"]
#     else:
#         if it_setup:
#             return ["it-thinking_guideline_higher_standards"]
#         else:
#             return ["thinking_guideline_higher_standards"]
    
def get_system_prompt(paragraph_id:int, group:str, prompt_type:str, few_shot:bool=False, shots:str=None, engineering:bool=False) -> str:
    if not few_shot:
        if prompt_type == "it-thinking_guideline":
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
"""
        elif prompt_type == "it-thinking_guideline_higher_standards":
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
"""
        elif prompt_type == 'it-german_vanilla':
            return f"""Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.
"""
        elif prompt_type == 'it-german_vanilla_expert':
            return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.
"""

        elif prompt_type == 'it-german_vanilla_expert_more_context':
            return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

Definition der Kategorien:
-`favour`: Die Haltung ist eindeutig und direkt positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
- `against`: Die Haltung ist eindeutig und direkt negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
- `neither`: Dies ist die Standardkategorie im Zweifelsfall. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt (siehe Zentrale Anweisungen).
"""
        
        elif prompt_type == 'it-german_more_context':
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
"""
        elif prompt_type == 'it-english_vanilla':
            # Vanilla prompt as in \citet{zhang_sentiment_2023}
            return f""" Please perform Stance Detection task. Given the Paragraph of a speech, assign a sentiment label expressed by the speaker towards "{group}" from [’against’, ’favor’, ’neither’]. Return label only without any other text.
"""
        elif prompt_type == 'it-german_vanilla_expert_more_context_cot':
            return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Zielgruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu.

Definition der Kategorien:
-`favour`: Die Haltung ist eindeutig und direkt positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
- `against`: Die Haltung ist eindeutig und direkt negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
- `neither`: Dies ist die Standardkategorie im Zweifelsfall. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt.

Bitte denke Schritt für Schritt nach und gebe im Anschluss deine Label in dem Format <stance>label</stance> aus.
"""

        elif prompt_type == "it-thinking_guideline_higher_standards_cot":
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
Bitte denke Schritt für Schritt nach und gebe im Anschluss deine Label in dem Format <stance>label</stance> aus.
"""

        elif prompt_type == "it-thinking_guideline_cot":
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
Bitte denke Schritt für Schritt nach und gebe im Anschluss deine Label in dem Format <stance>label</stance> aus.
"""
         
    elif few_shot:
        few_shot_string = build_few_shot_examples(paragraph_id, shots, engineering)
        if prompt_type == 'it-german_vanilla_expert':
            return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

{few_shot_string}
"""
        elif prompt_type == 'it-german_vanilla':
            return f"""Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

{few_shot_string}
"""
        elif prompt_type == 'it-german_vanilla_expert_more_context':
            return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

Definition der Kategorien:
-`favour`: Die Haltung ist eindeutig und direkt positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
- `against`: Die Haltung ist eindeutig und direkt negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
- `neither`: Dies ist die Standardkategorie im Zweifelsfall. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt (siehe Zentrale Anweisungen).

{few_shot_string}
"""
        elif prompt_type == "it-thinking_guideline_higher_standards":
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

{few_shot_string}
"""

        elif prompt_type == "it-thinking_guideline":
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

{few_shot_string}
"""

def get_test_batch(batch:pd.DataFrame, prompt_type:str, cot:bool=False, few_shot:bool=False, shots:str=None, engineering:bool=False) -> list[dict]:
    """ Takes a df of to be annotated data and a prompt type and returns formatted prompts.

    Args:
        batch (pd.DataFrame): Our to be annotated data from annotated_paragraphs table
        prompt_type (str): Corresponds to a prompt template
        few_shot (boo): Whether its a few_shot prompt or not
        shots (str): how many shots (e.g. 1,5 or 10)
        engineering (bool): indicates wheter we are testing on our engineering data or not

    Returns:
        list[dict]: Our prompt batch
    """
    prompt_batch = []
    for _, row in batch.iterrows():
        paragraph = row['inference_paragraph']
        paragraph_id = row['id']
        group = row['group_text']
        prompt = get_formatted_prompt(paragraph, group, prompt_type)
        promp = ""
        if few_shot:
            if shots is None:
                raise ValueError('Number of shots have to be specified!')
            else:
                prompt = get_formatted_few_shot_prompt(paragraph_id, shots, paragraph, group, prompt_type, engineering)
        elif cot:
            prompt = get_formatted_cot_prompt(paragraph, group, prompt_type)
        else:
            prompt = get_formatted_prompt(paragraph, group, prompt_type)
        # append formatted prompt to batch
        helper_dict = {
            "paragraph_id":paragraph_id,
            "prompt_type":prompt_type,
            "prompt":prompt
        }
        prompt_batch.append(helper_dict)

    return prompt_batch


def get_split_test_batch(batch:pd.DataFrame, prompt_type:str, few_shot:bool=False, shots:str=None, engineering:bool=False) -> list[dict]:
    #@todo rewrite this
    """ Takes a df of to be annotated data and a prompt type and returns formatted prompts.

    Args:
        batch (pd.DataFrame): Our to be annotated data from annotated_paragraphs table
        prompt_type (str): Corresponds to a prompt template
        few_shot (boo): Whether its a few_shot prompt or not
        shots (int): how many shots (e.g. 1,5 or 10)
        engineering (bool): indicates wheter we are testing on our engineering data or not

    Returns:
        list[dict]: Our prompt batch
    """
    prompt_batch = []
    for _, row in batch.iterrows():
        paragraph = row['inference_paragraph']
        paragraph_id = row['id']
        group = row['group_text']
        user_prompt = ""
        system_prompt = ""
        user_prompt = get_formatted_user_prompt(paragraph, group)
        if few_shot:
            system_prompt = get_system_prompt(paragraph_id, group, prompt_type, few_shot, shots, engineering)
        else:
            system_prompt = get_system_prompt(paragraph_id, group, prompt_type)
        
        # append formatted prompt to batch
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        helper_dict = {
            "paragraph_id":paragraph_id,
            "prompt_type":prompt_type,
            "message":message
        }
        prompt_batch.append(helper_dict)

    return prompt_batch

    

    
def get_engineering_data(sample_size:int=1):
    """ Function to extract a sample of our engineering data

    Args: 
        sample_size (int): The size of our sample

    Returns:
        None
    """
    con = connect_to_db(read_only=True)
    sql = """
        SELECT id, inference_paragraph, group_text 
        FROM engineering_data 
            JOIN annotated_paragraphs USING(id) 
        LIMIT ?
    """
    data = con.execute(sql, (sample_size,)).fetchdf()
    con.close()
    return data

def get_test_data():
    """ Function to extract our test data

    Args:
        None

    Returns:
        None
    """
    con = connect_to_db(read_only=True)
    sql = """
        SELECT id, inference_paragraph, group_text 
        FROM test_data 
            JOIN annotated_paragraphs USING(id);
    """
    data = con.execute(sql).fetchdf()
    con.close()
    return data

def build_few_shot_examples(test_paragraph_id:int, shots:int, engineering:bool=False) -> str:
    con = connect_to_db(read_only=True)
    few_show_sql = ""
    if engineering:
        few_show_sql = """
            SELECT * 
            FROM engineering_few_shot_examples fe
            JOIN annotated_paragraphs ap
                ON fe.sample_id = ap.id
            WHERE fe.eng_id = ? AND fe.k_shot = ?
        """
    else:
        few_show_sql = """
            SELECT * 
            FROM few_shot_examples fe 
            JOIN annotated_paragraphs ap
                ON fe.sample_id = ap.id
            WHERE fe.test_id = ? AND fe.k_shot = ?
        """
    few_shot_examples = con.execute(few_show_sql, (test_paragraph_id, shots)).fetchdf()
    example_string = ""
    
    for _, row in few_shot_examples.iterrows():
        paragraph = row['inference_paragraph']
        group = row['group_text']
        label = row['agreed_label']
        example_string += f"Textabschnitt: {paragraph}\nZielgruppe: {group}\nLabel: {label}\n\n"
        
    con.close()
    return example_string

def extract_stance_cot(answer: str) -> str:
    """
    Extracts the content from the <stance>-tags from an answer-string, with regex.

    Args:
        answer (str): The completes answer-string from the llm

    Returns:
        str: The extracted stance value or None if the format was wrong
    """
    # The regex pattern searches for the content between <stance> and </stance>.
    # (.*?) is a “non-greedy” capturing group that captures the text in between.
    # re.DOTALL ensures that the ‘.’ also includes line breaks.
    pattern = re.compile(r"<stance>(.*?)</stance>", re.DOTALL)
    
    # Search the text for the pattern
    match = pattern.search(answer)
    
    if match:
        # group(1) returns the content of the first capturing group
        # .strip() removes leading/trailing spaces or line breaks
        return match.group(1).strip()
    else:
        # Return None if the pattern was not found
        return None

def insert_prediction(engineering:bool, paragraph_id:int, model:str, prompt_type:str, technique:str, prediction:str, thoughts:str, thinking_process:str=None):
    """ This function is used to insert a models prediction into our db.

    Args:
        paragraph_id (int): The id of the predicted paragraph (corresponds to an annotated paragraph)
        model (str): The name of our model (e.g. 'gemini-2.5-pro')
        prompt_type (str): Corresponds to a prompt template
        technique (str): Prompting technique (e.g. 'zero-shot')
        prediction (str): The models predicted stance (e.g. 'favour')
        thinking_process (str): The output thinking process of reasoning models (e.g. for deepseek r1 everything in the <think> tag)
        thoughts (str): The thoughts of the CoT process
    """
    con = connect_to_db()
    con.begin()
    sql = ""
    if engineering:
        sql = """
            INSERT INTO engineering_predictions (id, model, prompt_type, technique, prediction, thinking_process, thoughts) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING --If a certain technique was already benchmarked, do nothing.
            """
    else:
        sql = """
            INSERT INTO predictions (id, model, prompt_type, technique, prediction, thinking_process, thoughts) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING --If a certain technique was already benchmarked, do nothing.
            """
    # Insert prediction into db
    try:
        con.execute(sql, (paragraph_id, model, prompt_type, technique, prediction, thinking_process, thoughts))
        con.commit()
        con.close()
    except ConstraintException:
        # If the model fails to provide output out of ['favour', 'against', 'neither']
        con.rollback()
        con.begin()
        con.execute(sql, (paragraph_id, model, prompt_type, technique, None, thinking_process, thoughts))
        con.commit()
        con.close()
        
def insert_batch(records:list[dict], engineering:bool):
    """ This function is used to insert a models prediction into our db.
    
    Args:
        records (list): the values, to be inserted into db
        engineering (bool): indicates wheter we are testing on our engineering data or not
    """
    con = connect_to_db()
    sql = ""
    if engineering:
        sql = """
            INSERT INTO engineering_predictions (id, model, prompt_type, technique, prediction, thinking_process, thoughts) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING --If a certain technique was already benchmarked, do nothing.
            """
    else:
        sql = """
            INSERT INTO predictions (id, model, prompt_type, technique, prediction, thinking_process, thoughts) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING --If a certain technique was already benchmarked, do nothing.
            """
    for record in records:
        con.begin()
        # Unpack the values from the dictionary for the current record
        paragraph_id = record.get("id")
        model = record.get("model")
        prompt_type = record.get("prompt_type") # This corresponds to the 'run' column
        technique = record.get("technique")
        prediction = record.get("prediction")
        thinking_process = record.get("thinking_process")
        thoughts = record.get("thoughts") # This corresponds to the 'parsed_stance' column
        # Insert prediction into db
        try:
            con.execute(sql, (paragraph_id, model, prompt_type, technique, prediction, thinking_process, thoughts))
            con.commit()
        except ConstraintException:
            # If the model fails to provide output out of ['favour', 'against', 'neither']
            con.rollback()
            con.begin()
            con.execute(sql, (paragraph_id, model, prompt_type, technique, None, thinking_process, thoughts))
            con.commit()
            
    con.close()


def get_formatted_user_prompt(paragraph:str, group:str) -> str:
    return f"""
Textabschnitt: {paragraph}
Zielgruppe: {group}
Label:
"""



    
def get_formatted_cot_prompt(paragraph:str, group:str, prompt_type:str) -> str:
    if prompt_type == 'german_vanilla_expert_more_context_cot':
        return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Zielgruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu.

Definition der Kategorien:
-`favour`: Die Haltung ist eindeutig und direkt positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
- `against`: Die Haltung ist eindeutig und direkt negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
- `neither`: Dies ist die Standardkategorie im Zweifelsfall. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt.

Bitte denke Schritt für Schritt nach und gebe im Anschluss deine Label in dem Format <stance>label</stance> aus.

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
"""

    elif prompt_type == "thinking_guideline_higher_standards_cot":
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
Bitte denke Schritt für Schritt nach und gebe im Anschluss deine Label in dem Format <stance>label</stance> aus.

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
"""

    elif prompt_type == "thinking_guideline_cot":
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
Bitte denke Schritt für Schritt nach und gebe im Anschluss deine Label in dem Format <stance>label</stance> aus.

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
"""

def get_formatted_few_shot_prompt(test_paragraph_id:int, shots:int, paragraph:str, group:str, prompt_type:str, engineering:bool) -> str:
    few_shot_string = build_few_shot_examples(test_paragraph_id, shots, engineering)
    if prompt_type == 'german_vanilla_expert':
        return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

{few_shot_string}
Textabschnitt: {paragraph}
Zielgruppe: {group}
Label:
        """
    elif prompt_type == 'german_vanilla':
        return f"""Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

    {few_shot_string}
    Textabschnitt: {paragraph}
    Zielgruppe: {group}
    Label:
"""
    elif prompt_type == 'german_vanilla_expert_more_context':
        return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

Definition der Kategorien:
-`favour`: Die Haltung ist eindeutig und direkt positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
- `against`: Die Haltung ist eindeutig und direkt negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
- `neither`: Dies ist die Standardkategorie im Zweifelsfall. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt (siehe Zentrale Anweisungen).

{few_shot_string}
Textabschnitt: {paragraph}
Zielgruppe: {group}
Label:
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

{few_shot_string}

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
"""

    elif prompt_type == "thinking_guideline":
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

{few_shot_string}

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
"""



    
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

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
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

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
"""
    elif prompt_type == 'german_vanilla':
        return f"""Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

Textabschnitt: {paragraph}
Zielgruppe: {group}
Label:
"""
    elif prompt_type == 'german_vanilla_expert':
        return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

Textabschnitt: {paragraph}
Zielgruppe: {group}
Label:
"""
    elif prompt_type == 'german_vanilla_expert_more_context':
        return f"""Du bist ein präziser Analyst für politische Sprache. Deine Aufgabe ist es, die Haltung (Stance) eines Sprechers gegenüber einer markierten Gruppe zu klassifizieren, basierend auf einem Textabschnitt.
Führe eine "Stance Detection" durch. Weise dem Sprecher im folgenden Textabschnitt eine Haltung (Stance) gegenüber "{group}" aus [’against’, ’favour’, ’neither’] zu. Gib nur das Label ohne weiteren Text zurück.

Definition der Kategorien:
-`favour`: Die Haltung ist eindeutig und direkt positiv. Der Sprecher lobt die Gruppe, verteidigt sie, fordert etwas zu ihren Gunsten oder gibt explizit an, für ihre Interessen einzutreten (z.B. "wir kämpfen für diese Gruppe").
- `against`: Die Haltung ist eindeutig und direkt negativ. Der Sprecher kritisiert, verurteilt oder warnt vor der Gruppe oder macht sie für ein Problem verantwortlich.
- `neither`: Dies ist die Standardkategorie im Zweifelsfall. Wähle sie, wenn die Haltung neutral, ambivalent oder unklar ist, oder wenn der Sprecher die Gruppe nur sachlich erwähnt (siehe Zentrale Anweisungen).

Textabschnitt: {paragraph}
Zielgruppe: {group}
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

Textabschnitt: {paragraph}
Zielgruppe: {group}
Antwort:
"""
    elif prompt_type == 'english_vanilla':
        # Vanilla prompt as in \citet{zhang_sentiment_2023}
        return f""" Please perform Stance Detection task. Given the Paragraph of a speech, assign a sentiment label expressed by the speaker towards "{group}" from [’against’, ’favor’, ’neither’]. Return label only without any other text.

Paragraph: {paragraph}
Target: {group}
Label:
"""


if __name__ == '__main__':
    print("hello")
    print(build_few_shot_examples(test_paragraph_id=6924663, shots='5-shot', engineering=False))