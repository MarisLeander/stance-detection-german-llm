#%%
import duckdb
import xml.etree.ElementTree as ET
from tqdm import tqdm
# Import classifier
#from ipynb.fs.full.group_classifier import predict_batch
from group_classifier import predict_batch
from datasets.utils.logging import disable_progress_bar
# Disables progress bar of .map function
disable_progress_bar()
print("Import successful!!")
#%%
 # Connect to sql database
con = duckdb.connect(database='../data/database/german-parliament.duckdb', read_only=False)
#%%
def preprocess_speech(speech: tuple) -> tuple[int, list[dict[str, str]]]:
    """The filtering for skipping_president_remarks is only necessare for periods >= 19 because of the "new" format provided by the bundestag. 
    For periods <19 the method just extracs all the 'p' tags

    Args:
        speech (tuple): A tuple containing the speech data

    Returns:
        A tuple containing the speech ID and a list of paragraphs with their text content.
    """
    speech_id = speech[0]
    text_content = speech[5]

    # Parse xml from string will throw ParseError if not parseable
    root = ET.fromstring(text_content)
    
    paragraphs_text = []
    # skipping_president_remarks is used to detect interruptions of the President
    skipping_president_remarks = False

    # Iterate over all direct children of the root <rede> element
    for element in root:
        # 1. Check if we need to STOP skipping
        if skipping_president_remarks:
            if element.tag == 'p' and element.attrib.get("klasse") == "redner":
                # We know that the interruption of the president ended and the speech continues after this tag
                skipping_president_remarks = False
                continue 
            elif element.tag == 'p':
                # Do nothing -> president is speaking
                continue
            elif element.tag == 'kommentar':
                # Do nothing -> comment from the crowd, which we are not interested in
                continue
            else:
                print(f"Detected unexpected tag: {element.tag}")

        # 2. Check if we need to START skipping (President speaks)
        if element.tag == 'name':
            # Check if the text content of the name tag indicates a presiding
            name_text = "".join(element.itertext()).strip() # Gets all text within <name>, including children
            # State all titles which will get ignored
            president_titles = ["Präsidentin", "Präsident", "Vizepräsidentin", "Vizepräsident"]
            if name_text and any(title in name_text for title in president_titles):
                skipping_president_remarks = True
                continue # Move to the next element, don't process this <name> tag as a paragraph

        # 3. Process <p> tags
        if element.tag == 'p':
            # Filter out speaker information paragraphs
            if element.attrib.get("klasse") == "redner":
                continue
            else:
                # Actual text content of speech!
                # Get text of the paragraph and remove potential irrelevant whitespaces
                p_text = element.text.strip() if element.text else ""
                
                item = {"paragraphs": p_text}
                paragraphs_text.append(item)
        
        # Other tags like <kommentar> are implicitly skipped as they are not 'p'

    return (speech_id, paragraphs_text)

#%%
def create_paragraphs_classified_table(reset_db:bool=False):
    """
    Creates a table for classified paragraphs in the database.

    Args:
        reset_db (bool): If True, drops the table if it exists before creating it.
                         Defaults to False.
    Returns:
        None
    """
    if reset_db:
        con.execute("DROP TABLE IF EXISTS group_mention")
        con.execute("DROP SEQUENCE IF EXISTS group_mention_id_seq")

    # Create a sequence for the primary key
    con.execute("CREATE SEQUENCE IF NOT EXISTS group_mention_id_seq START 1;")

    con.execute("""
        CREATE TABLE IF NOT EXISTS group_mention (
            id INTEGER DEFAULT nextval('group_mention_id_seq') PRIMARY KEY,
            paragraph_no INTEGER, -- If its 0, its the first paragraph of the speech, 1 for the second, etc.
            speech_id VARCHAR NOT NULL REFERENCES speech(id),
            paragraph VARCHAR NOT NULL,
            group_text VARCHAR NOT NULL, -- This is the group mention, e.g. die Mitglieder der SPD-Fraktion
            label VARCHAR(15) NOT NULL,
        )
    """)
    con.commit()
#%%
def smart_join(tokens):
    """
    Joins a list of word tokens into a single string, handling punctuation
    and sub-word prefixes ('##') correctly.

    Args:
        tokens (list[str]): A list of word tokens, which may include sub-word tokens prefixed with '##'.

    Returns:
        str: A single string with tokens joined together, ensuring proper spacing around punctuation.
    """
    result = []
    # Define punctuation that should not have a preceding space
    no_space_before = {',', '.', '?', '!', ';', ':', ')'}

    for i, token in enumerate(tokens):
        # If it's the very first token, a punctuation mark, or a sub-word,
        # don't add a leading space.
        if i > 0 and token not in no_space_before and not token.startswith('##'):
            result.append(' ')

        # Append the token itself, removing any '##' prefixes
        result.append(token.replace('##', ''))

    return "".join(result).replace(' - ', '-') # Removes the space before and after a hyphen (Bindestrich)

def extract_groups(paragraph:list[tuple[str,str]]) -> list[tuple[str, str]]:
    """ Extracts group mention along with their labels from a paragraph. It groups tokens by their entity labels to get the full mention.
    If a mention is broken e.g it does not start with a 'B-' label, it will be filtered.

    Args:
        paragraph (list[tuple[str,str]]): A list of tuples containing tokens and their corresponding labels.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains the entity label and a list of (token, label) pairs for that entity, which contain the full mention.
    """
    # This is a set of special tokens that should be ignored in the grouping process. -> adjust it if necessary
    SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
    # empty list for groups of paragraph
    groups = []
    # This is a temporary list to hold the current group mention
    group_tmp = []
    # This is a flag to indicate if we are currently inside a group mention
    group_started = False
    entity = "" # This hold the current entity. e.g. EOPOL for B-EOPOL
    for token, label in paragraph:
        # If token is a special token like [CLS] skip it
        if token in SPECIAL:
            continue
        # Check for begin of group
        elif label.startswith("B-"):
            group_started = True
            entity = label[2:]
            if group_tmp:
                groups.append((entity, group_tmp))
                group_tmp = [] # New Group will begin
            # Append new beginning label
            group_tmp.append((token, label))
        # It is checked that 1) we have an inside label and 2) There was a B- label before!
        elif label.startswith("I-") and group_started:
            # Then we check if the entity matches
            if label[2:] != entity:
                # print(f"Current label: {label[2:]} doesn't match beginning label: {entity}") @todo handle this as error log
                # Break current group because of the miss-label
                group_started = False
            else:
            # If all tests hold, we append the token and its label to the current group
                group_tmp.append((token, label))
        elif label == 'O':
            # An 'O' Tag is always outside. Thus if we scan one, it means that the current group is over
            if group_tmp:
                groups.append((entity, group_tmp))
                group_tmp = [] # New Group will begin
            group_started = False
        else:
            pass
            # print(f"Filtered faulty classification: ({token}, {label})") @todo handle this as error log

    # Flush last word
    if group_tmp:
        groups.append((entity, group_tmp))

    return groups

def insert_paragraph(speech_id:int, index:int, entity:str, group_clean_text:str, paragraph:str):
    """
    Inserts a classified paragraph into the database.

    Args:
        speech_id (int): The ID of the speech.
        index (int): The index of the paragraph in the speech. 0 for the first paragraph, 1 for the second, etc.
        entity (str): The entity label of the paragraph. For example, 'EOPOL' for B-EOPOL.
        group_clean_text (str): The cleaned text of the paragraph.
        paragraph (str): The original paragraph text.

    Returns:
        None
    """
    con.execute("""
        INSERT INTO group_mention (paragraph_no, speech_id, paragraph, group_text, label)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING; -- If the paragraph already exists, do nothing
    """, (index, speech_id, paragraph, group_clean_text, entity))
    con.commit()

def insert_group_mention(speech_id:str, index:int, groups:list[tuple[str,list[tuple[str,str]]]], paragraph:str):
    """Inserts classified paragraphs into the database.

    Args:
        speech_id (str): The ID of the speech.
        index (int): The index of the paragraph in the speech. 0 for the first paragraph, 1 for the second, etc.
        groups (list[tuple[str,list[tuple[str,str]]]]): A list of tuples containing the entity and a list of the token, label pairs for each group.
        paragraph (str): The original paragraph text.

    Returns:
        None
    """
    for group in groups:
        entity, raw_tokens = group
        tokens = [item[0] for item in raw_tokens]
        group_clean_text = smart_join(tokens)
        # print(f"{entity} -> {group_clean_text}")
        insert_paragraph(speech_id, index, entity, group_clean_text, paragraph)


def process_speech(speech_id:str, paragraphs:list[dict[str, list[str]]]):
    """Processes a speech by classifying its paragraphs (extracting group mention) and inserting them into the database.

    Args:
        speech_id (str): The ID of the speech.
        paragraphs (list[dict[str, list[str]]]): The list of paragraphs, each represented as a dictionary with a 'paragraphs' key containing the text.
    """
    group_mention = predict_batch(paragraphs)
    for index, p in enumerate(group_mention):
        # print(p)
        groups = extract_groups(p)
        insert_group_mention(speech_id, index, groups, paragraphs[index].get('paragraphs'))

def extract_speech() -> tuple:
    #@todo speed up this query -> extract more than 1 entry at a time :)
    """Extracts a random speech from the database that has not been processed yet.

    Args:
        None

    Returns:
        tuple: A tuple containing the speech data, including its ID, title, date, and text content.
    """
    sql = """
        SELECT *
        FROM speech
        WHERE position NOT IN ('Präsidentin', 'Vizepräsidentin', 'Vizepräsident', 'Präsident')
              AND id NOT IN (SELECT speech_id FROM group_mention) -- check that speech wasn't already processed
        ORDER BY RANDOM()
        LIMIT 1
        """
    return con.execute(sql).fetchone()
#%%
def main():
    """ Main function to process speeches and extract group mentions.
    """
    create_paragraphs_classified_table(reset_db=False)
    for i in tqdm(range(2000), desc='Processing:'):
        speech = extract_speech()
        speech_id, paragraphs = preprocess_speech(speech)
        process_speech(speech_id, paragraphs)

if __name__ == "__main__":
    main()
    con.close()
#%%
