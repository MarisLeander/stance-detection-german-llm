#%%
import duckdb as db
from pathlib import Path
import pandas as pd
import csv 
from datetime import datetime
import argparse 
#%%

def connect_db() -> db.DuckDBPyConnection:
    """ Connect to the database

    Args:
        None
        
    Returns:
         duckdb.DuckDBPyConnection: Connection to our database.

    """
    # Get path to db
    home_dir = Path.home()
    path = home_dir / "stance-detection-german-llm" / "data" / "database" / "german-parliament.duckdb"
    
     # Connect to sql database
    return db.connect(database=path, read_only=False)
#%%
def save_csv_to_user_data_folder(df_to_save: pd.DataFrame, filename: str):
    """
    Builds a path to a specific folder within the user's home directory,
    creates the folder if it doesn't exist, and saves a DataFrame there.

    Args:
        df_to_save (pd.DataFrame): The Pandas DataFrame to be saved.
        filename (str): The name for the output CSV file (e.g., "results.csv").
    """
    try:
        # Get the user's home directory
        home_dir = Path.home()

        # Join path
        target_dir = home_dir / "stance-detection-german-llm" / "data" / "annotation_data"
        
        # Create the directory if it doesn't exist
        # `parents=True` creates any missing parent folders.
        # `exist_ok=True` prevents an error if the folder already exists.
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the full path to the output file
        output_filepath = target_dir / filename
        
        # Save the DataFrame to the constructed path
        df_to_save.to_csv(output_filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        print(f"Successfully saved file to: {output_filepath}")

    except Exception as e:
        print(f"An error occurred: {e}")

def build_to_be_annotated_data(target_file:str, con:db.DuckDBPyConnection, label_limit:int=10):
    """" Builds a CSV file with paragraphs and group mentions to be annotated. It selects random paragraphs from the group_mention table, for which the group mention is not in the ignore list.
        It creates a new column 'formatted_paragraph' that contains the paragraph with the group mention highlighted.

    Args:
        target_file (str): The path to the CSV file where the data will be saved.
        con (db.DuckDBPyConnection): Connection to our database.

    Returns:
        None
    """
    print(f"Saving data to: {target_file}...")
    #@ todo not ignore groups, rather normalize them!!!!!
    # Get all labels
    excluded_labels = ['GPE', 'EPOWN']
    labels = con.execute(f"SELECT DISTINCT(label) FROM group_mention WHERE label NOT IN {excluded_labels}").fetchdf().label.tolist()
    print(f"Current labels in Database: {labels}")
    # List for dataframes of each label
    dataframes = []
    for label in labels:
        sql = f"""
            SELECT *
            FROM group_mention g 
                JOIN speech s 
                ON g.speech_id = s.id 
            WHERE g.label = '{label}' 
                AND LENGTH(paragraph) <= 3000 
            ORDER BY RANDOM() 
            LIMIT ?
        """
        dataframes.append(con.execute(sql,(label_limit,)).fetchdf())
        
    # Concat the entries for all labels
    data = pd.concat(dataframes, ignore_index=True)
    save_csv_to_user_data_folder(data, target_file)


def main():
    """ Main function to execute the data preparation for annotation. """

    parser = argparse.ArgumentParser(description="Extract annotation data of mentions.")
    
    # Add an argument for the limit.
    # We define its name (--limit), type (int), a default value, and a help message.
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10, 
        help="The maximum number of each label to extract."
    )
    # Get passed args
    args = parser.parse_args()
    
    con = connect_db()
    con.begin() # start transaction
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    build_to_be_annotated_data(f'annotation_data_{timestamp}.csv', con, label_limit=args.limit)
    con.commit() # commit transaction
    con.close()
if __name__ == "__main__":
    main()

