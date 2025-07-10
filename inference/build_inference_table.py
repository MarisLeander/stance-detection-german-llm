import duckdb as db
from pathlib import Path
import argparse

def connect_to_db(db_path: str) -> db.DuckDBPyConnection:
    """
    Connect to a DuckDB database.

    Args:
        db_path (str): The path to the DuckDB database file.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the DuckDB database.
    """
    return db.connect(database=db_path, read_only=False)


def create_inference_table(con:db.DuckDBPyConnection, reset_predictions:bool=False) -> None:
    """
    Build the table with the predictions from our models.

    Args:
        con (db.DuckDBPyConnection): The connection to the DuckDB database.
        reset_db (bool): Indicates whether the database will be reset or not

    Returns:
        None
    """

    if reset_predictions:
        con.execute("DROP TABLE IF EXISTS predictions;")

    sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER NOT NULL REFERENCES engineering_data(id), --CHANGE THIS TO test_data!!!!
            model VARCHAR NOT NULL,
            prompt_type VARCHAR,
            technique VARCHAR NOT NULL, --e.g. 0-shot, k-shot, CoT, etc.
            prediction VARCHAR(8) CHECK (prediction IN ('favour', 'against', 'neither')),
            thoughts VARCHAR, -- direct output from an api or <think><\\think} from a reasoning model
            thinking_process VARCHAR, -- thinking in the output for CoT prompting
            PRIMARY KEY(id, model, prompt_type, technique)
        )
    """

    con.execute(sql)


def main():
    """ Main function is used to get db and creation of the predictions table.
    """
    # Get db path and connection
    home_dir = Path.home()
    db_path = home_dir / "stance-detection-german-llm" / "data" / "database" / "german-parliament.duckdb"
    
    con = connect_to_db(db_path)


    # build argparser, to pass information in cli whether to reset db or not when starting the script
    parser = argparse.ArgumentParser(description="Process annotated data.")

    parser.add_argument(
        "--reset_predictions",
        action='store_true', # This is the key change
        help="If this flag is present, all tables will be reset."
    )

    args = parser.parse_args()

    # Create table    
    create_inference_table(con, args.reset_predictions)


if __name__ == "__main__":
    main()










