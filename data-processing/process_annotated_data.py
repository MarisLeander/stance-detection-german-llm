import duckdb as db
from pathlib import Path
import argparse
import pandas as pd
import json
import math
from tqdm import tqdm 

def connect_to_db(db_path: str) -> db.DuckDBPyConnection:
    """
    Connect to a DuckDB database.

    Args:
        db_path (str): The path to the DuckDB database file.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the DuckDB database.
    """
    return db.connect(database=db_path, read_only=False)

def create_data_tables(con:db.DuckDBPyConnection, reset_db:bool=False) -> None:
    """
    Build the table with our annotated paragraphs, aswell as our engineering, test splits.

    Args:
        con (db.DuckDBPyConnection): The connection to the DuckDB database.
        reset_db (bool): Indicates if the whole database will be reset

    Returns:
        None
    """
    if reset_db:
        # con.execute("DROP TABLE IF EXISTS predictions") --only reset this if we want to wipe our predictions table. Handle with caution!
        con.execute("DROP TABLE IF EXISTS few_shot_examples")
        con.execute("DROP TABLE IF EXISTS annotations")
        con.execute("DROP TABLE IF EXISTS engineering_data")
        con.execute("DROP TABLE IF EXISTS test_data")
        con.execute("DROP TABLE IF EXISTS annotated_paragraphs")

    sql = """
        CREATE TABLE IF NOT EXISTS annotated_paragraphs (
            id INTEGER PRIMARY KEY REFERENCES group_mention(id),
            group_text VARCHAR NOT NULL,
            inference_paragraph VARCHAR NOT NULL,
            adjusted_span BOOLEAN NOT NULL,
            agreed_label VARCHAR(16) CHECK (agreed_label IN ('favour', 'against', 'neither', 'not a group')) --The label which is agreed on between the annotators
            );
    """
    con.execute(sql)
    


def create_annotation_table(con:db.DuckDBPyConnection, reset_annotations:bool=False) -> None:
    """
    Creates our tables to insert our annotations from our annotators into.

    Args:
        con (db.DuckDBPyConnection): The connection to the DuckDB database.
        reset_db (bool): Indicates if the annotations will be reset

    Returns:
        None
    """
    if reset_annotations:
        con.execute("DROP TABLE IF EXISTS annotations")

    sql = """
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER NOT NULL,
            annotator VARCHAR(32) NOT NULL,
            annotated_paragraph_id INTEGER NOT NULL REFERENCES annotated_paragraphs(id),
            stance VARCHAR(16) CHECK (stance IN ('favour', 'against', 'neither', 'not a group')),
            PRIMARY KEY(id, annotator)
        );
    """
    con.execute(sql)
    
    
def adjust_inference_para(paragraph:str, group_span_adjustment:list[dict]) -> str:
    # remove the span tags
    clean_paragraph = paragraph.replace("<span>", "").replace("</span>", "")
    # get placement of adjusted group span

    offsets = group_span_adjustment[0].get('globalOffsets')
    start = offsets.get("start")
    end = offsets.get("end")


    # input validation
    # ensures start and end are valid integers and in the correct order.
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(clean_paragraph)):
        print(f"Warning: Invalid offsets provided. start={start}, end={end}, text_length={len(clean_paragraph)}")
        # Return the original text if offsets are invalid to prevent errors.
        return None

    # slice string, based on offsets to build new inference paragraph
    before_span = clean_paragraph[:start]
    span_content = clean_paragraph[start:end]
    after_span = clean_paragraph[end:]

    # construct the new string with the tags wrapped around the middle part.
    return f"{before_span}<span>{span_content}</span>{after_span}"

def process_primary_annotations(path:str, con:db.DuckDBPyConnection) -> None:
    """ Processes annotations from the primary annotator. This will be used, to
        build our annotated_paragraphs table. That means only the inference paragraphs are used / adjusted.

    Args:
        path (str): The path to the annotation file.
        con (db.DuckDBPyConnection): the database connection

    Returns:
        None
    """
    # Read annotated data
    annotation_data = pd.read_csv(path)
    # Iterate over each annotated entry and add it to db
    for index, row in annotation_data.iterrows():
        group_text = row['group_text']
        paragraph_id = row['id']
        inference_paragraph = row['inference_paragraph']
        group_span_adjustments = row['answer']
        adjusted_span = False

        if not pd.isna(group_span_adjustments):
            adjusted_span = True
            # Convert list string into list
            group_span_adjustments = json.loads(group_span_adjustments)
            # If our adjustments are not NA it means, that the group span was adjusted!
            inference_paragraph = adjust_inference_para(inference_paragraph, group_span_adjustments)
            # Adjust group text, since the group span was adjusted
            group_text = group_span_adjustments[0].get('text')
            
        # Insert annotated paragraph into db
        sql = """
            INSERT INTO annotated_paragraphs (id, group_text, inference_paragraph, adjusted_span)
                 VALUES (?, ?, ?, ?)
              """
        con.execute(sql, (paragraph_id, group_text, inference_paragraph, adjusted_span))

def process_annotations(path:str, annotator:str, con:db.DuckDBPyConnection) -> None:
    """ Processes annotations from an annotator. This will be used, to fill the annotations table, where annotations from each annotator are stored
    
    Args:
        path (str): The path to the annotation file.
        annotator (str): The name of the annotator
        con (db.DuckDBPyConnection): The database connection

    Returns:
        None
        
    """

    # Read annotated data
    annotation_data = pd.read_csv(path)

    for index, row in annotation_data.iterrows():
        id = index
        annotator = annotator.lower()
        paragraph_id = row['id']
        stance = row['stance_annotation'].lower()
        sql = "INSERT INTO annotations (id, annotator, annotated_paragraph_id, stance) VALUES (?, ?, ?, ?);"
        con.execute(sql, (id, annotator, paragraph_id, stance))

def agree_on_labels(con:db.DuckDBPyConnection) -> None:
    """ 
    Sets the agreed labels for all annotated paragraphs in the following way:
        1. If the annotators agree on a label, this will be the agreed label.
        2. If the annotators don't agree on a label, and one annotator label 'not a group' the agreed label will be set to 'not a group'.
        3. In the remaining case the agreed on label will be set to 'neither' since the paragraph is ambiguous.

    Args:
        con (db.DuckDBPyConnection): Our database connection
        
    Returns:
        None
    
    """
    
    # --- 1. Get all labels where the annotators agree on and save them into our annotated_paragraphs ---
    agreed_labels_sql = """
        -- First, calculate the total number of unique annotators in the entire table
        WITH AnnotatorTotal AS (
            SELECT COUNT(DISTINCT annotator) AS total_annotators
            FROM annotations
        )
        
        -- Secont, find the paragraphs that meet both agreement and completeness criteria
        SELECT
            annotated_paragraph_id,
            MIN(stance) AS agreed_stance -- Shows the stance everyone agreed on
        FROM
            annotations
        GROUP BY
            annotated_paragraph_id
        HAVING
            -- Condition 1: All annotators agree on the stance
            COUNT(DISTINCT stance) = 1
            
            -- Condition 2: The number of annotators for this paragraph equals the total number of annotators (check if all annotators annotated the paragraph)
            AND COUNT(annotator) = (SELECT total_annotators FROM AnnotatorTotal);
    """
    agreed_labels = con.execute(agreed_labels_sql).fetchdf()

    for index, row in agreed_labels.iterrows():
        # Insert agreed stance into annotated_paragraphs table
        sql = "UPDATE annotated_paragraphs SET agreed_label = ? WHERE id = ?;"
        con.execute(sql, (row['agreed_stance'], row['annotated_paragraph_id']))
            
    # --- 2. Get all labels where the annotators don't share agreement but one of the annotators labeled "not a group" ---
    not_groups_disagreement_sql = """
        -- First, get the total number of unique annotators for the completeness check
        WITH AnnotatorTotal AS (
            SELECT COUNT(DISTINCT annotator) AS total_annotators
            FROM annotations
        )
        
        -- Second, find the paragraphs that meet all three conditions (below)
        SELECT
            annotated_paragraph_id,
            -- Aggregates the stances to shows the specific conflict
            string_agg(annotator || ': ' || stance, ', ') AS conflicting_stances
        FROM
            annotations
        GROUP BY
            annotated_paragraph_id
        HAVING
            -- Condition 1: All annotators have labeled this paragraph
            COUNT(annotator) = (SELECT total_annotators FROM AnnotatorTotal)
        
            -- Condition 2: The annotators do not agree (there is more than one unique stance)
            AND COUNT(DISTINCT stance) > 1
        
            -- Condition 3: At least one of the stances in the group is 'not a group'
            AND bool_or(stance = 'not a group');
    """
    disagreement_data = con.execute(not_groups_disagreement_sql).fetchdf()

    for index, row in disagreement_data.iterrows():
        sql = "UPDATE annotated_paragraphs SET agreed_label = 'not a group' WHERE id = ?;"
        con.execute(sql, (row['annotated_paragraph_id'],))
        
    # --- 3. The only remaining case is, that the annotators didn't agree on a label, which is explicitly not 'not a group' ---
    sql = """
        UPDATE annotated_paragraphs 
        SET agreed_label = 'neither' 
        WHERE id IN (SELECT id 
                     FROM annotated_paragraphs 
                     WHERE agreed_label IS NULL);
        """
    # Set all remaining entries to 'neither', since the annotators couldn't agree on a label (except if one labeled 'not a group', which is handled before)
    con.execute(sql)


def create_test_engineering_split(con:db.DuckDBPyConnection) -> None:
    """
    Creates our engineering, test splits.

    Args:
        con (db.DuckDBPyConnection): The connection to the DuckDB database.
        reset_db (bool): Indicates if the database will be reset

    Returns:
        None
    """

    con.execute("CREATE TABLE IF NOT EXISTS engineering_data (id INTEGER PRIMARY KEY REFERENCES annotated_paragraphs(id));")
    # Get count of not a group lables
    overall_count = con.execute("SELECT COUNT(*) FROM annotated_paragraphs").fetchone()[0]
    nag_count = con.execute("SELECT COUNT(*) FROM annotated_paragraphs WHERE agreed_label = 'not a group'").fetchone()[0]
    remaining_para = overall_count - nag_count
    engineering_amount = math.floor((remaining_para) * 0.1)
    test_amount = math.ceil((remaining_para) * 0.9)
    print(f"Overall {nag_count} paragraphs have been labeled 'not a group'. \n Engineering/test split will be {engineering_amount}/{test_amount} of the remaining {remaining_para} paragraphs")
    sql = f"""
        INSERT INTO engineering_data (id)
        SELECT id
        FROM annotated_paragraphs
        WHERE agreed_label != 'not a group'
        USING SAMPLE reservoir({engineering_amount} ROWS) REPEATABLE (42);
        """
    con.execute(sql)

    con.execute("CREATE TABLE IF NOT EXISTS test_data (id INTEGER PRIMARY KEY REFERENCES annotated_paragraphs(id));")

    sql = """
        INSERT INTO test_data (id)
        SELECT id
        FROM annotated_paragraphs
        WHERE agreed_label != 'not a group'
            AND id NOT IN (SELECT id from engineering_data);
        """
    con.execute(sql)


def create_few_shot_table(con:db.DuckDBPyConnection) -> None:
    """
    Extracts sample for few-shot approaches

    Args:
        con (db.DuckDBPyConnection): The connection to the DuckDB database.
    """
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS few_shot_examples (
            test_id INTEGER NOT NULL REFERENCES test_data(id),
            k_shot VARCHAR NOT NULL CHECK(k_shot IN('1-shot','5-shot','10-shot')),
            sample_id INTEGER NOT NULL REFERENCES engineering_data(id)
        );
    """
    con.execute(create_table_sql)

    # We only take examples where our annotators agreed on a label!
    test_data = con.execute("SELECT id FROM test_data").fetchdf()
    shots = [1,5,10]
    
    for shot in tqdm(shots, desc="Building few_shot_examples table..."):
        for _, row in test_data.iterrows():
            test_id = int(row['id'])
            k_shot = f"{shot}-shot"
            sql = f"""
                WITH LabelersAnnotations AS (
                        SELECT
                            maris.annotated_paragraph_id AS annotated_paragraph_id,
                            maris.stance AS maris_stance,
                            harriet.stance AS harriet_stance
                        FROM
                            annotations AS maris
                        JOIN
                            annotations AS harriet ON maris.annotated_paragraph_id = harriet.annotated_paragraph_id
                        WHERE
                            maris.annotator = 'maris'
                            AND harriet.annotator = 'harriet'
                )
                SELECT eg.id 
                FROM engineering_data td 
                    JOIN LabelersAnnotations la 
                    ON eg.id = la.annotated_paragraph_id
                USING SAMPLE reservoir({shot} ROWS) REPEATABLE (42);
            """
            # sql = f"""
            #     SELECT id
            #     FROM engineering_data
            #     USING SAMPLE reservoir({shot} ROWS) REPEATABLE (42);
            #     """
            sample_ids = con.execute(sql).fetchdf()
            for _, row in sample_ids.iterrows():
                sample_id = int(row['id'])
                
                con.execute("INSERT INTO few_shot_examples (test_id, k_shot, sample_id) VALUES (?, ?, ?)", (test_id, k_shot, sample_id))
            
            


def main():
    """ Main function to process annotated data.

    Args:
        None

    Returns:
        None
    """

    # Get path to db
    home_dir = Path.home()
    db_path = home_dir / "stance-detection-german-llm" / "data" / "database" / "german-parliament.duckdb"
    # Get db connection
    con = connect_to_db(db_path)



    # build argparser, to pass information whether to reset db or not when starting the script
    parser = argparse.ArgumentParser(description="Process annotated data.")

    parser.add_argument(
        "--reset_db",
        action='store_true', # This is the key change
        help="If this flag is present, all tables will be reset."
    )
    parser.add_argument(
        "--reset_annotations",
        action='store_true', # This is the key change
        help="If this flag is present, the annotations table will be reset."
    )
    args = parser.parse_args()
    
    con.begin()
    if args.reset_db:
        create_data_tables(con, reset_db=True)
        create_annotation_table(con, reset_annotations=True)
    elif args.reset_annotations:
        create_annotation_table(con, reset_annotations=True)
    con.commit()
    
    print("Start processing of annotations...")
 
    # ---- Build annotated_paragraphs table ----
    con.begin()
    path =  home_dir / "stance-detection-german-llm" / "data" / "annotated_data"/ "maris-processed-2025-06-30-14-57-790f3829.csv"
    process_primary_annotations(path, con)
    con.commit()

    # --- Build annotations table for each annotator ---
    annotator_files = [("maris-processed-2025-06-30-14-57-790f3829.csv", "maris"),('harriet-processed-2025-07-06-16-42-e62bcc7f.csv', 'harriet')]

    for file, name in annotator_files:
        path =  home_dir / "stance-detection-german-llm" / "data" / "annotated_data"/ file
        con.begin()
        process_annotations(path, name, con)
        con.commit()

    # --- Extract agreed on labels and insert neither for labels which aren't agreed on.
    con.begin()
    agree_on_labels(con)
    con.commit()
    
    # --- Build test and engineering split ---
    # @todo ignore not a group lables
    if args.reset_db:
        con.begin()
        create_test_engineering_split(con) # If the whole db is reset, we need to rebuild the engineering / test split
        con.commit()

    # --- Build few-shot id's table ----
    if args.reset_db:
        con.begin()
        create_few_shot_table(con)
        con.commit()
    
    con.close()
    print("Finished processing")



if __name__ == "__main__":
    main()
