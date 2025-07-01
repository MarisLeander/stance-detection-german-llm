#%%
import duckdb as db
from pathlib import Path
import pandas as pd
import csv 
from datetime import datetime
import argparse 
from tqdm import tqdm
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

def get_sample_quantity(con:db.DuckDBPyConnection) -> pd.DataFrame:

    sql = """
        WITH FilteredData AS (
            SELECT label, paragraph
            FROM group_mention
            WHERE label NOT IN ('GPOWN', 'GPE')
              AND LENGTH(paragraph) <= 2000
        ),
        
        -- get the punished total count
        PunishedTotal AS (
            SELECT SUM(
                CASE
                    -- Apply 1/40th weight for EOPOL and EPPOL
                    WHEN label IN ('EOPOL', 'EPPOL') THEN 0.025
                    -- Apply 1/5th weight for PFUNK
                    WHEN label = 'PFUNK' THEN 0.2
                    -- All other labels get full weight
                    ELSE 1.0
                END
            ) AS punished_total_count
            FROM FilteredData
        )
        
        -- get the final sample number for each label
        SELECT
            fd.label,
            ROUND(
                10 + (
                    (
                        CASE
                            -- 1/40th punishment to the count for EOPOL and EPPOL
                            WHEN fd.label IN ('EOPOL', 'EPPOL') THEN (COUNT(*) * 0.025)
                            -- 1/5th punishment to the count for PFUNK
                            WHEN fd.label = 'PFUNK' THEN (COUNT(*) * 0.2)
                            ELSE COUNT(*)
                        END
                    ) * 1.0 / pt.punished_total_count
                ) * 750
            )::INTEGER AS sampleNumber
        FROM
            FilteredData AS fd,
            PunishedTotal AS pt
        GROUP BY
            fd.label, 
            pt.punished_total_count
        ORDER BY
            sampleNumber DESC;
    """

    return con.execute(sql).fetchdf()

def build_to_be_annotated_data(target_file:str, con:db.DuckDBPyConnection):
    """" Builds a CSV file with paragraphs and group mentions to be annotated. It selects random paragraphs from the group_mention table, for which the group mention is not in the ignore list.
        It creates a new column 'formatted_paragraph' that contains the paragraph with the group mention highlighted.

    Args:
        target_file (str): The path to the CSV file where the data will be saved.
        con (db.DuckDBPyConnection): Connection to our database.

    Returns:
        None
    """
    print(f"Saving data to: {target_file}...")

    sample_dataframe = get_sample_quantity(con)

    # List of dataframes for each label
    dataframes = []
    
    for index, row in tqdm(sample_dataframe.iterrows(), total=len(sample_dataframe), desc="Gathering Samples"):
        # We use a dedicated sampling method (reservoir sampling) to get a good random distribution
        # See: https://duckdb.org/docs/stable/sql/samples.html
        sql = f"""
            -- Filter table for label
            WITH FilteredResults AS (
                SELECT
                    *
                FROM
                    group_mention g
                JOIN
                    speech s ON g.speech_id = s.id
                WHERE
                    g.label = ?
                    AND LENGTH(g.paragraph) <= 3000
            )
            -- Sampling from the pre-filtered results
            SELECT *
            FROM FilteredResults
            USING SAMPLE ({row.sampleNumber} ROWS)
            REPEATABLE (100); --random seed
        """
        
        dataframes.append(con.execute(sql,(row.label,)).fetchdf())
        
    # Concat the entries for all labels
    data = pd.concat(dataframes, ignore_index=True)
    data = data.sample(frac=1) # Shuffle dataframe, so that we have labels in random order when annotating
    save_csv_to_user_data_folder(data, target_file)


def main():
    """ Main function to execute the data preparation for annotation. """
    con = connect_db()
    con.begin() # start transaction
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    build_to_be_annotated_data(f'annotation_data_{timestamp}.csv', con)
    
    con.commit() # commit transaction
    con.close()
    
if __name__ == "__main__":
    main()

