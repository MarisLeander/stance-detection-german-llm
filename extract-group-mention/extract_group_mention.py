# ------
# Disables progress bar of .map function
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
# -------
# Since the imports take quite some time we Signal if they are done
print("Starting Worker")
from collections import Counter
import argparse 
from datetime import datetime
import duckdb
import xml.etree.ElementTree as ET
import pandas as pd
import time
import multiprocessing 
from tqdm import tqdm
# Import classifier
from group_classifier import GroupClassifier
print("Worker is setup")
#%%

def log_statistics( overall_stats: Counter, elapsed_minutes: float, processed_speeches: int, log_file: str = "statistics.txt"):
    """
    Formats the final statistics, prints them to the console,
    and appends them to a log file.

    Args:
        overall_stats (Counter): The counter object with processing statistics.
        elapsed_minutes (float): The total runtime in minutes.
        processed_speeches (int): The number of speeches processed (from the --limit arg).
        log_file (str): The path to the log file.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    # Create a header for the log entry
    header = f"--- Statistics for run on {timestamp} ---\n"
    
    # Format each key-value pair from the Counter for readability
    stats_lines = [f"  - {key:<35}: {value:,}" for key, value in overall_stats.items()]
    formatted_stats = "\n".join(stats_lines)
    
    # Format the summary lines. <35 means left alignment with a reserved width of 35 chars (padding)
    summary = (
        f"\n  ----------------------------------------\n"
        f"  - {'Overall extracted paragraphs':<35}: {overall_stats.total():,}\n"
        f"  - {'Total Speeches Processed':<35}: {processed_speeches:,}\n"
        f"  - {'Execution Time (minutes)':<35}: {round(elapsed_minutes, 2)}"
    )
    
    output_string = header + formatted_stats + summary + "\n\n"

    print("\n" + output_string)

    # Append the formatted string to the log file
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(output_string)
        print(f"Statistics successfully appended to {log_file}")
    except IOError as e:
        print(f"Error: Could not write to log file {log_file}. Reason: {e}")



def preprocess_speech(speech: tuple) -> tuple[int, list[dict[str, str]], Counter]:
    """The filtering for skipping_president_remarks is only necessare for periods >= 19 because of the "new" format provided by the bundestag. 
    For periods <19 the method just extracs all the 'p' tags

    Args:
        speech (tuple): A tuple containing the speech data

    Returns:
        A tuple containing the speech ID and a list of paragraphs with their text content.
    """
    speech_id = speech.id
    text_content = speech.content

    # Parse xml from string will throw ParseError if not parseable
    root = ET.fromstring(text_content)
    num_of_important_paragraphs = 0
    num_of_redner_information = 0
    num_of_comments = 0
    num_of_president_paragraphs = 0
    num_of_mislabeled_paragraphs = 0
    
    paragraphs_text = []
    # skipping_president_remarks is used to detect interruptions of the President
    skipping_president_remarks = False

    # Iterate over all direct children of the root <rede> element
    for element in root:
        # 1. Check if we need to STOP skipping
        if skipping_president_remarks:
            if element.tag == 'p' and element.attrib.get("klasse") == "redner":
                num_of_redner_information += 1
                # We know that the interruption of the president ended and the speech continues after this tag
                skipping_president_remarks = False
                continue 
            elif element.tag == 'p':
                num_of_president_paragraphs += 1
                # Do nothing -> president is speaking
                continue
            elif element.tag == 'kommentar':
                num_of_comments += 1
                # Do nothing -> comment from the crowd, which we are not interested in
                continue
            else:
                num_of_mislabeled_paragraphs += 1
                print(f"Detected unexpected tag: {element.tag}")

        # 2. Check if we need to START skipping (President speaks)
        if element.tag == 'name':
            num_of_redner_information += 1
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
                num_of_redner_information += 1
                continue
            else:
                num_of_important_paragraphs += 1
                # Actual text content of speech!
                # Get text of the paragraph and remove potential irrelevant whitespaces
                p_text = element.text.strip() if element.text else ""
                
                item = {"paragraphs": p_text}
                paragraphs_text.append(item)
        # 4. Process any other tags
        elif element.tag == 'kommentar' or  element.tag == 'stage':
            # Comments from the crowd are labeled as 'stage' in periods < 19
            num_of_comments += 1
        elif element.tag == 'speaker':
            num_of_redner_information += 1
        else:
            num_of_mislabeled_paragraphs += 1
            print(f"Detected unexpected tag: {element.tag}")
        
        
    # Create the counter
    stats_dict = Counter({
        "num_of_important_paragraphs": num_of_important_paragraphs,
        "num_of_redner_information": num_of_redner_information,
        "num_of_comments": num_of_comments,
        "num_of_president_paragraphs": num_of_president_paragraphs,
        "num_of_mislabeled_paragraphs": num_of_mislabeled_paragraphs,
    })

    return (speech_id, paragraphs_text, stats_dict)

#%%
def create_tables(con:duckdb.DuckDBPyConnection, reset_db:bool=False):
    """
    Creates a table for classified paragraphs in the database.

    Args:
        con (duckdb.DuckDBPyConnection): Connection to our database
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
            group_text VARCHAR NOT NULL, -- This is the group mention, e.g. die Mitglieder der SPD-Fraktion
            label VARCHAR(15) NOT NULL,
            paragraph VARCHAR NOT NULL,
            annotation_paragraph VARCHAR NOT NULL, 
            inference_paragraph VARCHAR NOT NULL
        )
    """)
    con.commit()
    
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
    special_tokens_to_exclude = {"[UNK]","[PAD]","[CLS]"} # Set of tokens to ignore

    for i, token in enumerate(tokens):
        if token in special_tokens_to_exclude:
            continue
        # If it's the very first token, a punctuation mark, or a sub-word,
        # don't add a leading space.
        if i > 0 and token not in no_space_before and not token.startswith('##'):
            result.append(' ')

        # Append the token itself, removing any '##' prefixes
        result.append(token.replace('##', ''))

    return "".join(result).replace(' - ', '-') # Removes the space before and after a hyphen (Bindestrich)

def format_text(before_text:str, mention_text:str, after_text:str) -> tuple[str,str]:
    """
    Creates a styled HTML paragraph and highlights the exact group mention
    specified by its start and end indices.

    This version uses precise string slicing instead of regular expressions to ensure
    only the specific occurrence is highlighted.

    Args:
        before_text (str): The content of the paragraph in front of the group mention
        mention_text (str): The exact text of the group mention.
        after_text (str): The content of the paragraph after the group mention
    Returns:
        tuple: A tuple, containing an HTML string for the annotaters, aswell as a lowkey tagged string, for LLM inference
    """

    # Style for the highlight span itself
    highlight_style = "background-color: #FFFDE7; border-radius: 4px; padding: 1px 5px; box-shadow: inset 0 -2px 0 #FFD600;"
    
    # Base style for the entire paragraph container
    paragraph_style = "font-family: 'Inter', sans-serif; font-size: 16px; line-height: 1.7; color: #333;"

    # Construction of the highlighted HTML
    # Wrap the sliced mention text with the highlight style
    highlighted_mention_html = f' <span style="{highlight_style}">{mention_text}</span> '
    
    # Re-assemble the paragraph and wrap it in a styled div
    final_content = f"{before_text}{highlighted_mention_html}{after_text}"
    final_styled_annotator_paragraph = f'<div style="{paragraph_style}">{final_content}</div>'
    final_styled_inference_paragraph = f"{before_text} <span>{mention_text}</span> {after_text}"

    return (" ".join(final_styled_annotator_paragraph.split()), " ".join(final_styled_inference_paragraph.split())) # We get rid of duplicate whitespaces / tabs / returns


    

def extract_groups(paragraph:list[tuple[str,str]]) -> list[tuple[str, str, str, str]]:
    """ Extracts group mention along with their labels from a paragraph. It groups tokens by their entity labels to get the full mention.
    If a mention is broken e.g it does not start with a 'B-' label, it will be filtered.

    Args:
        paragraph (list[tuple[str,str]]): A list of tuples containing tokens and their corresponding labels.

    Returns:
        list[tuple[str, str, str, str]]: A list of tuples where each tuple contains the entity label and a list of (token, label) pairs for that entity, as well
                                         as styled paragraphs for further annotation and inference (annotation_paragraph, inference_paragraph)
    """
    groups = []
    current_group_tokens = []
    current_entity = ""
    # Get the text in front of the group mention.
    text_before_mention = ""

    for index, (token, label) in enumerate(paragraph):
        # Check if a new group starts
        if label.startswith("B-"):
            # If there's a pending group, save it first
            if current_group_tokens:
                clean_text = smart_join(current_group_tokens)
                # Get the current token, and the sucessing tokens
                following_tokens = [t[0] for t in paragraph[index:]]
                # Convert tokens into string
                text_after_mention = smart_join(following_tokens)
                annotation_paragraph, inference_paragraph = format_text(text_before_mention, clean_text, text_after_mention)
                groups.append((current_entity, clean_text, annotation_paragraph, inference_paragraph))
            
            # Start the new group
            current_entity = label[2:]
            current_group_tokens = [token]
            # Get the tokens before the group mention
            previous_tokens = [t[0] for t in paragraph[:index]]
            # Convert tokens into string
            text_before_mention = smart_join(previous_tokens)
        # Check if the current token continues the existing group
        elif label.startswith("I-") and label[2:] == current_entity:
            current_group_tokens.append(token)
        # If the token is 'O' or an invalid 'I-' tag, the group ends
        else:
            if current_group_tokens:
                clean_text = smart_join(current_group_tokens)
                # Get the current token, and the sucessing tokens
                following_tokens = [t[0] for t in paragraph[index:]]
                # Convert tokens into string
                text_after_mention = smart_join(following_tokens)
                annotation_paragraph, inference_paragraph = format_text(text_before_mention, clean_text, text_after_mention)
                groups.append((current_entity, clean_text, annotation_paragraph, inference_paragraph))
            
            # Reset for the next potential group
            current_group_tokens = []
            current_entity = ""
            
    # After the loop, save any pending group
    if current_group_tokens:
        clean_text = smart_join(current_group_tokens)
        # Get the current token, and the sucessing tokens
        following_tokens = [t[0] for t in paragraph[index:]]
        # Convert tokens into string
        text_after_mention = smart_join(following_tokens)
        annotation_paragraph, inference_paragraph = format_text(text_before_mention, clean_text, text_after_mention)
        groups.append((current_entity, clean_text, annotation_paragraph, inference_paragraph))

    return groups


def extract_speeches(con:duckdb.DuckDBPyConnection, limit:int = 1000) -> pd.DataFrame:
    #@todo speed up this query -> extract more than 1 entry at a time :)
    """Extracts a random speech from the database that has not been processed yet.

    Args:
        con (duckdb.DuckDBPyConnection): Connection to our database.

    Returns:
        tuple: A tuple containing the speech data, including its ID, title, date, and text content.
    """
    sql = """
        SELECT *
        FROM speech
        WHERE position NOT IN ('Präsidentin', 'Vizepräsidentin', 'Vizepräsident', 'Präsident')
              OR position IS NULL
              AND id NOT IN (SELECT distinct(speech_id) FROM group_mention) -- check that speech wasn't already processed
        ORDER BY RANDOM()
        LIMIT ?
        """
    return con.execute(sql, (limit,)).fetchdf()
#%%
def main():
    """
    Main function to efficiently process all speeches in a batch.
    """

    parser = argparse.ArgumentParser(description="Process speeches to find group mentions.")
    
    # Add an argument for the limit.
    # We define its name (--limit), type (int), a default value, and a help message.
    parser.add_argument(
        "--limit", 
        type=int, 
        default=1000, 
        help="The maximum number of speeches to process."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="The amount of workers (threads) to extract group mention"
    )

    parser.add_argument(
        "--reset_db",
        type=bool,
        default=False,
        help="Decised whether the group_mention table should get reset, or not"
    )

    # Parse the arguments passed from the command line
    args = parser.parse_args()


    
    total_start_time = time.time()
    # Connect to sql database
    con = duckdb.connect(database='stance-detection-german-llm/data/database/german-parliament.duckdb', read_only=False)
    
    print("--- Starting Batch Speech Processing ---")
    
    # Load ingthe classifier model once at the start.
    model_path = "stance-detection-german-llm/models/bert-base-german-cased-finetuned-MOPE-L3_Run_3_Epochs_29"
    classifier = GroupClassifier(model_dir=model_path)
    
    # Build the group mention and paragraph table
    create_tables(con, reset_db=args.reset_db)
    
    # Fetch all speeches from the database.
    speeches_df = extract_speeches(con, limit=args.limit)
    
    all_paragraphs_text = []
    # This list will store metadata to remember where each paragraph came from.
    # Each item will be a tuple: (speech_id, original_paragraph_index)
    paragraph_metadata = [] 
    
    overall_statistics = Counter({
        "num_of_important_paragraphs": 0,
        "num_of_redner_information": 0,
        "num_of_comments": 0,
        "num_of_president_paragraphs": 0,
        "num_of_mislabeled_paragraphs": 0,
    })
    
    print("\nPreprocessing speeches and collecting all paragraphs...")
    for _, row in tqdm(speeches_df.iterrows(), total=len(speeches_df), desc="Gathering Paragraphs"):
        speech_id, paragraphs_list_of_dicts, item_statistics = preprocess_speech(row)
        # Add counter to our stats
        overall_statistics += item_statistics
        for i, para_dict in enumerate(paragraphs_list_of_dicts):
            # Assuming the text is in para_dict['paragraphs']
            paragraph_text = para_dict.get('paragraphs')
            if paragraph_text:
                all_paragraphs_text.append(paragraph_text)
                paragraph_metadata.append((speech_id, i)) # Save the origin
    
    if not all_paragraphs_text:
        print("No paragraphs to process. Exiting.")
        return

    # Efficient batched inference
    # Processes all paragraphs in one go on the GPU.
    # Use a large batch size to maximize A100 utilization.
    print(f"\nStarting batch prediction on {len(all_paragraphs_text)} paragraphs...")
    start_time = time.time()
    
    all_predictions = classifier.predict(all_paragraphs_text, batch_size=256, num_workers=args.workers)
    
    end_time = time.time()
    print(f"--- Prediction finished in {end_time - start_time:.2f} seconds ---")

    #@todo do this in a method maybe
    # Collect all records before inserting
    print("\nExtracting groups and preparing for bulk insert...")
    records_to_insert = []
    #@ todo perform smart join!
    for i, metadata in enumerate(tqdm(paragraph_metadata, desc="Preparing Records")):
        speech_id, paragraph_index = metadata
        predicted_tokens_and_labels = all_predictions[i]
        # original_paragraph_text = all_paragraphs_text[i]
        
        groups = extract_groups(predicted_tokens_and_labels)
        tokens = [item[0] for item in predicted_tokens_and_labels]
        group_joined_text = smart_join(tokens)
        for entity, group_text, annotation_paragraph, inference_paragraph in groups:
            # Append a dictionary for easy conversion to a DataFrame
            records_to_insert.append({
                "paragraph_no": paragraph_index,
                "speech_id": speech_id,
                "group_text": group_text,
                "label": entity,
                "paragraph": group_joined_text,
                "annotation_paragraph": annotation_paragraph,
                "inference_paragraph": inference_paragraph,
            })
    # Perform a single, massive bulk insert
    if records_to_insert:
        print(f"\nStarting bulk insert of {len(records_to_insert):,} records...")
        start_time = time.time()

        # Convert the list of dictionaries to a Pandas DataFrame
        df_to_insert = pd.DataFrame(records_to_insert)
        
        # DuckDB is highly optimized to ingest DataFrames this way.
        con.execute("""
            INSERT INTO group_mention (paragraph_no, speech_id, group_text, label, paragraph, annotation_paragraph, inference_paragraph)
            SELECT paragraph_no, speech_id, group_text, label, paragraph, annotation_paragraph, inference_paragraph FROM df_to_insert
        """)
        
        end_time = time.time()

        
    else:
        print("\nNo group mentions found to insert.")
        
    print("\n--- Processing complete! ---")
    
    # Calculate final elapsed time
    elapsed_time_minutes = (time.time() - total_start_time) / 60
    
    # Call the new function to print and log the final stats
    log_statistics(
        overall_stats=overall_statistics,
        elapsed_minutes=elapsed_time_minutes,
        processed_speeches=args.limit 
    )
    con.close()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # This can happen if the context is already set, which is fine.
        pass
    main()

