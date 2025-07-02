import duckdb as db
from pathlib import Path
import argparse
import pandas as pd
import json

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
        con.execute("DROP TABLE IF EXISTS annotations")
        con.execute("DROP TABLE IF EXISTS engineering_data")
        con.execute("DROP TABLE IF EXISTS test_data")
        con.execute("DROP TABLE IF EXISTS annotated_paragraphs")

    sql = """
        CREATE TABLE IF NOT EXISTS annotated_paragraphs (
            id INTEGER PRIMARY KEY REFERENCES group_mention(id),
            group_text VARCHAR NOT NULL,
            inference_paragraph VARCHAR NOT NULL,
            adjusted_span BOOLEAN NOT NULL
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

    sql = """
        INSERT INTO engineering_data (id)
        SELECT id
        FROM annotated_paragraphs
        USING SAMPLE reservoir(100 ROWS) REPEATABLE (42);
        """
    con.execute(sql)

    con.execute("CREATE TABLE IF NOT EXISTS test_data (id INTEGER PRIMARY KEY REFERENCES annotated_paragraphs(id));")

    sql = """
        INSERT INTO test_data (id)
        SELECT id
        FROM annotated_paragraphs
        WHERE id NOT IN (SELECT id from engineering_data);
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
    path =  home_dir / "stance-detection-german-llm" / "data" / "annotated_data"/ "maris-2025-06-30-14-57-790f3829.csv"
    process_primary_annotations(path, con)
    con.commit()

    # --- Build annotations table for each annotator ---
    annotator_files = [("maris-2025-06-30-14-57-790f3829.csv", "maris"),('harriet_tmp-2025-07-01-13-33-c16b5183.csv', 'harriet_tmp')]

    for file, name in annotator_files:
        path =  home_dir / "stance-detection-german-llm" / "data" / "annotated_data"/ file
        con.begin()
        process_annotations(path, name, con)
        con.commit()

    
    if args.reset_db:
        con.begin()
        create_test_engineering_split(con) # If the whole db is reset, we need to rebuild the engineering / test split
        con.commit()
    
    con.close()
    print("Finished processing")



if __name__ == "__main__":
    main()
