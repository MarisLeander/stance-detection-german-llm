import duckdb as db
import pandas as pd
from pathlib import Path

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

def create_evaluation_table(con:db.DuckDBPyConnection, reset_db:bool=True):
    con.begin()
    if reset_db:
        con.execute("DROP TABLE IF EXISTS eval_matrix;")
        con.execute("DROP TABLE IF EXISTS label_f1;")
        con.execute("DROP TABLE IF EXISTS model_evaluation;")
        con.execute("DROP SEQUENCE IF EXISTS config_id_seq;")
        
    # Create a sequence for config IDs to ensure unique IDs for each config. It's like an auto-incrementing primary key.
    con.execute("CREATE SEQUENCE IF NOT EXISTS config_id_seq START 1;")
    
    create_overall_table_sql = """
        CREATE TABLE IF NOT EXISTS model_evaluation (
            config_id INTEGER DEFAULT nextval('config_id_seq') PRIMARY KEY,
            model VARCHAR NOT NULL,
            prompt_type VARCHAR NOT NULL,
            technique VARCHAR NOT NULL, --e.g. 0-shot, k-shot, CoT, etc.
            failure_rate FLOAT NOT NULL,
            loose_macro_f1 FLOAT NOT NULL,
            strict_macro_f1 FLOAT NOT NULL,
            UNIQUE (model, prompt_type, technique)
        );
    """
    create_label_table_sql = """
        CREATE TABLE IF NOT EXISTS label_f1 (
            config_id INTEGER NOT NULL REFERENCES model_evaluation(config_id),
            label VARCHAR(8) NOT NULL CHECK (label IN ('favour', 'against', 'neither')),
            loose_f1 FLOAT NOT NULL,
            strict_f1 FLOAT NOT NULL,
            PRIMARY KEY(config_id, label)
        );
    """

    create_matrix_table_sql = """
        CREATE TABLE IF NOT EXISTS eval_matrix (
            config_id INTEGER NOT NULL REFERENCES model_evaluation(config_id),
            label VARCHAR(8) NOT NULL,
            eval_type VARCHAR(8) NOT NULL CHECK (eval_type IN ('strict', 'loose')),
            tp INTEGER NOT NULL,
            fp INTEGER NOT NULL,
            tn INTEGER NOT NULL,
            fn INTEGER NOT NULL,
            PRIMARY KEY (config_id, label, eval_type)
        );
    """
    con.execute(create_overall_table_sql)
    con.execute(create_label_table_sql)
    con.execute(create_matrix_table_sql)
    con.commit()
    
def calculate_f1(tp:int, fp:int, tn:int, fn:int) -> float:
    """
    Calculates precision, recall, and F1 score from TP, FP, TN, and FN,
    with handling for division-by-zero errors.

    Args:
        tp (int): True Positives
        fp (int): False Positives
        tn (int): True Negatives
        fn (int): False Negatives

    Returns:
        float: The calculated F1 score
    """
    # If the model made no positive predictions, precision is 0.
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # If there were no actual positive instances, recall is 0.
    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # If both precision and recall are 0, the F1 score is 0.
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return f1_score

def macro_f1_formula(f1_scores:list[str,int]) -> int:
     # Calculates the sum of the f1 scores
    total_sum = sum(item for item in f1_scores)
    
    # Divides the sum by the total number of items.
    macro_f1 = total_sum / len(f1_scores)
    return macro_f1

def calculate_strict_f1(predictions_df:pd.DataFrame, label:str, con:db.DuckDBPyConnection) -> float:
    # Get ids of all predictions for current label
    label_pred_ids = predictions_df[predictions_df['prediction'] == label].id.tolist()
    # Get ids of all prediction for the other labels
    neg_label_pred_ids = predictions_df[(predictions_df['prediction'] != label) & (predictions_df['prediction'].notna())].id.tolist()
    # Get true pos, false pos, etc.
    tp = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label = ?", (tuple(label_pred_ids), label)).fetchone()[0]
    fp = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label <> ?", (tuple(label_pred_ids), label)).fetchone()[0]
    tn = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label <> ?", (tuple(neg_label_pred_ids), label)).fetchone()[0]
    fn = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label = ?", (tuple(neg_label_pred_ids), label)).fetchone()[0]
    # print(f"TP: {tp}")
    # print(f"FP: {fp}")
    # print(f"TN: {tn}")
    # print(f"FN: {fn}")
    f1_score = calculate_f1(tp, fp, tn, fn)
    return (f1_score, {"tp":tp, "fp":fp, "tn":tn, "fn":fn})

def calculate_loose_f1(predictions_df:pd.DataFrame, label:str, con:db.DuckDBPyConnection) -> float:
    # Get ids of all predictions for current label
    label_pred_ids = predictions_df[predictions_df['prediction'] == label].id.tolist()
    # Get ids of all prediction for the other labels
    neg_label_pred_ids = predictions_df[(predictions_df['prediction'] != label) & (predictions_df['prediction'].notna())].id.tolist()
    # Create view to compare annotations of the annotators
    view_sql = """
        CREATE VIEW LabelersAnnotations AS (
            SELECT
                maris.annotated_paragraph_id,
                maris.stance AS maris_stance,
                harriet.stance AS harriet_stance
            FROM
                annotations AS maris
            JOIN
                annotations AS harriet ON maris.annotated_paragraph_id = harriet.annotated_paragraph_id
            WHERE
                maris.annotator = 'maris'
                AND harriet.annotator = 'harriet'
    );
    """
    con.execute(view_sql)

    # The Ground Truth is 'favour' if harriet_stance == 'favour' OR maris_stance == 'favour'.
    # The Ground Truth is 'not favour' if harriet_stance != 'favour' AND maris_stance != 'favour'.
    
    # Either one of the annotators stance annotations has to agree on the predictions of the model
    # e.g. model predicts 'favour' AND the Ground Truth is 'favour'.
    tp_sql = """
        SELECT COUNT(DISTINCT annotated_paragraph_id)
        FROM 
            LabelersAnnotations
        WHERE 
            annotated_paragraph_id IN ? 
            AND (harriet_stance = ? OR maris_stance = ?);
    """
    tp = con.execute(tp_sql, (tuple(label_pred_ids), label, label)).fetchone()[0]
    # None of the annotators stance annotations agrees on the predictions of the model
    # e.g. model predicts 'favour' AND the Ground Truth is 'not favour'.
    fp_sql = """
        SELECT COUNT(DISTINCT annotated_paragraph_id)
        FROM 
            LabelersAnnotations
        WHERE 
            annotated_paragraph_id IN ? 
            AND harriet_stance <> ? 
            AND maris_stance <> ?;
    """
    fp = con.execute(fp_sql, (tuple(label_pred_ids), label, label)).fetchone()[0]

    # Ground truth is not the label and the model doesn't predict the label
    # e.g. model predicts 'not favour' AND the Ground Truth is 'not favour'.
    tn_sql = """
        SELECT COUNT(DISTINCT annotated_paragraph_id)
        FROM 
            LabelersAnnotations
        WHERE 
            annotated_paragraph_id IN ? 
            AND harriet_stance <> ? 
            AND maris_stance <> ?;
    """
    tn = con.execute(tn_sql, (tuple(neg_label_pred_ids), label, label)).fetchone()[0]

    # Ground trouth is the label, i.e. one of the annotators annotated it, but the model annotated something else
    # e.g. model predicts 'not favour' (i.e., 'against' or 'neither') AND the Ground Truth is 'favour'.
    fn_sql = """
        SELECT COUNT(DISTINCT annotated_paragraph_id)
        FROM 
            LabelersAnnotations
        WHERE 
            annotated_paragraph_id IN ? 
            AND (harriet_stance = ? OR maris_stance = ?);
    """
    fn = con.execute(fn_sql, (tuple(neg_label_pred_ids), label, label)).fetchone()[0]
    f1_score = calculate_f1(tp, fp, tn, fn)
    # Drop our view
    con.execute("DROP VIEW LabelersAnnotations;")
    return (f1_score, {"tp":tp, "fp":fp, "tn":tn, "fn":fn})

def insert_macro_f1(
    predictions_df:pd.DataFrame,
    con:db.DuckDBPyConnection,
    failure_rate:float,
    strict_macro_f1:float,
    loose_macro_f1:float
):
    model = predictions_df.loc[0, 'model']
    prompt_type = predictions_df.loc[0, 'prompt_type']
    technique = predictions_df.loc[0, 'technique']
    con.begin()
    insert_sql = "INSERT INTO model_evaluation (model, prompt_type, technique, failure_rate, loose_macro_f1, strict_macro_f1) VALUES (?, ?, ?, ?, ?, ?);"
    con.execute(insert_sql, (model, prompt_type, technique, failure_rate, loose_macro_f1, strict_macro_f1))
    con.commit()


def get_config_id(predictions_df:pd.DataFrame, con:db.DuckDBPyConnection) -> int:
    """ Gets the config id for a given (model, prompt_type, technique) combination.
    """
    model = predictions_df.loc[0, 'model']
    prompt_type = predictions_df.loc[0, 'prompt_type']
    technique = predictions_df.loc[0, 'technique']
    sql = "SELECT config_id FROM model_evaluation WHERE model = ? AND prompt_type = ? AND technique = ?"
    return con.execute(sql, (model, prompt_type, technique)).fetchone()[0]
    
def insert_label_f1(
    predictions_df:pd.DataFrame, 
    con:db.DuckDBPyConnection,
    label:str, 
    strict_f1_score:float, 
    loose_f1_score:float
):
    config_id = get_config_id(predictions_df, con)
    insert_sql = "INSERT INTO label_f1 (config_id, label, loose_f1, strict_f1) VALUES (?, ?, ?, ?)"
    con.begin()
    con.execute(insert_sql, (config_id, label, loose_f1_score, strict_f1_score))
    con.commit()

def insert_eval_matrix(
    predictions_df: pd.DataFrame, 
    con: db.DuckDBPyConnection,
    label: str, 
    strict_matrix_dict: dict[str, int],
    loose_matrix_dict: dict[str, int]
):
    """
    Efficiently inserts the strict and loose evaluation matrices for a given label
    into the database using a single executemany call.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing prediction run info.
        con (db.DuckDBPyConnection): An active connection to the DuckDB database.
        label (str): The specific label these metrics apply to (e.g., 'favour').
        strict_matrix_dict (dict): A dictionary with keys 'tp', 'fp', 'tn', 'fn'.
        loose_matrix_dict (dict): A dictionary with keys 'tp', 'fp', 'tn', 'fn'.
    """
    # --- Step 1: Get the configuration ID for this run ---
    config_id = get_config_id(predictions_df, con)

    # --- Step 2: Prepare the data for bulk insertion ---
    # Create a list of tuples, where each tuple represents a row to be inserted.
    records_to_insert = [
        (
            config_id,
            label,
            'strict', # The evaluation type
            strict_matrix_dict.get('tp', 0),
            strict_matrix_dict.get('fp', 0),
            strict_matrix_dict.get('tn', 0),
            strict_matrix_dict.get('fn', 0)
        ),
        (
            config_id,
            label,
            'loose', # The evaluation type
            loose_matrix_dict.get('tp', 0),
            loose_matrix_dict.get('fp', 0),
            loose_matrix_dict.get('tn', 0),
            loose_matrix_dict.get('fn', 0)
        )
    ]

    # --- Step 3: Define the SQL and execute in a single batch ---
    # This is much more efficient than looping and calling execute() multiple times.
    insert_sql = """
        INSERT INTO eval_matrix (config_id, label, eval_type, tp, fp, tn, fn)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    
    con.executemany(insert_sql, records_to_insert)
    
def calculate_macro_f1(predictions_df:pd.DataFrame, failure_rate:int, con:db.DuckDBPyConnection):
    labels = ['favour', 'against', 'neither']
    strict_f1_per_class = []
    loose_f1_per_class = []
    for label in labels:
        # For the strict f1 score the model has to predict the agreed on label, which is neither if the annotators labelled different labels.
        strict_f1_score, strict_matrix_dict = calculate_strict_f1(predictions_df, label, con)
        strict_f1_per_class.append({"label":label, 
                                   "strict_f1_score":strict_f1_score, 
                                   "matrix_dict":strict_matrix_dict})
        # For the loose f1 score, the model has only to predict one of the labels each annotator labelled.
        loose_f1_score, loose_matrix_dict = calculate_loose_f1(predictions_df, label, con)
        loose_f1_per_class.append({"label":label, 
                                   "loose_f1_score":loose_f1_score, 
                                   "matrix_dict":loose_matrix_dict})
    # Get all of our strict_f1 scores per label, to calculate the macro_f1
    strict_scores_list = [item['strict_f1_score'] for item in strict_f1_per_class]
    strict_macro_f1 = macro_f1_formula(strict_scores_list)
    # Get all of our loose_f1 scores per label, to calculate the macro_f1
    loose_scores_list = [item['loose_f1_score'] for item in loose_f1_per_class]
    loose_macro_f1 = macro_f1_formula(loose_scores_list)
    # Insert our macro f1 scores
    insert_macro_f1(predictions_df, con, failure_rate, strict_macro_f1, loose_macro_f1)
    
    for i in range (0, 3):
        # Get for each label the strict evaluation information for the db
        strict_label = strict_f1_per_class[i].get("label")
        strict_f1_score = strict_f1_per_class[i].get("strict_f1_score")
        strict_matrix_dict = strict_f1_per_class[i].get("matrix_dict")
        # Get for each label the loose evaluation information for the db
        loose_label = loose_f1_per_class[i].get("label")
        loose_f1_score = loose_f1_per_class[i].get("loose_f1_score")
        loose_matrix_dict = loose_f1_per_class[i].get("matrix_dict")
        # Check if the labels are the same
        if strict_label != loose_label:
            raise ValueError('Labels are not the same!')
        else:
            # Insert eval. into db
            insert_label_f1(predictions_df, con, strict_label, strict_f1_score, loose_f1_score)
            insert_eval_matrix(predictions_df, con, strict_label, strict_matrix_dict, loose_matrix_dict)

    

def calculate_failure_rate(model:str, technique:str, prompt_type:str, con:db.DuckDBPyConnection):
    """ Calculates in how many cases the model failed to provide a correct formatted output

    Args:
        model (str): The name of our model (e.g. 'gemini-2.5-pro')
        technique (str): Prompting technique (e.g. 'zero-shot')
        prompt_type (str): Corresponds to a prompt template
        con (db.DuckDBPyConnection): The connection to our db
    """
    total_preds_sql = "SELECT COUNT (DISTINCT id) FROM predictions WHERE model = ? AND technique = ? AND prompt_type = ?;"
    total_preds = con.execute(total_preds_sql, (model, technique, prompt_type)).fetchone()[0]
    null_preds_sql = "SELECT COUNT (DISTINCT id) FROM predictions WHERE model = ? AND technique = ? AND prompt_type = ? AND prediction IS NULL;"
    null_preds = con.execute(null_preds_sql, (model, technique, prompt_type)).fetchone()[0]
    failure_rate = round(((null_preds / total_preds) * 100), 2)
    return failure_rate
    

def evaluate_predictions(con:db.DuckDBPyConnection):
    models = con.execute("SELECT DISTINCT model FROM predictions;").fetchall() 
    techniques = con.execute("SELECT DISTINCT technique FROM predictions;").fetchall()
    for model in models:
        for technique in techniques: 
            prompt_types = con.execute("SELECT DISTINCT prompt_type FROM predictions WHERE model = ? AND technique = ?;", (model[0], technique[0])).fetchall()
            for prompt_type in prompt_types:
                pred_sql = "SELECT * FROM predictions WHERE model = ? AND technique = ? AND prompt_type = ?;"
                predictions_df = con.execute(pred_sql, (model[0], technique[0], prompt_type[0])).fetchdf()
                failure_rate = calculate_failure_rate(model[0], technique[0], prompt_type[0], con)
                calculate_macro_f1(predictions_df, failure_rate, con)
                
                


def main():
    con = connect_to_db()
    create_evaluation_table(con)
    evaluate_predictions(con)
    con.close()



if __name__ == "__main__":
    main()