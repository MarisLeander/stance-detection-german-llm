import argparse
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


def create_evaluation_table(con: db.DuckDBPyConnection, reset_db: bool = True):
    """
    Sets up the database tables for storing evaluation results.
    
    MODIFICATION:
    - Added 'evaluation_scope' column to 'model_evaluation' table.
    - Updated UNIQUE constraint to include 'evaluation_scope'.
    """
    con.begin()
    if reset_db:
        con.execute("DROP TABLE IF EXISTS eval_matrix;")
        con.execute("DROP TABLE IF EXISTS label_f1;")
        con.execute("DROP TABLE IF EXISTS model_evaluation;")
        con.execute("DROP SEQUENCE IF EXISTS config_id_seq;")

    con.execute("CREATE SEQUENCE IF NOT EXISTS config_id_seq START 1;")

    create_overall_table_sql = """
        CREATE TABLE IF NOT EXISTS model_evaluation (
            config_id INTEGER DEFAULT nextval('config_id_seq') PRIMARY KEY,
            model VARCHAR NOT NULL,
            prompt_type VARCHAR NOT NULL,
            technique VARCHAR NOT NULL,
            evaluation_scope VARCHAR NOT NULL, -- ADDED: 'overall' or 'agreed_only'
            failure_rate FLOAT NOT NULL,
            strict_macro_f1 FLOAT NOT NULL,
            strict_micro_f1 FLOAT NOT NULL,
            loose_macro_f1 FLOAT NOT NULL,
            loose_micro_f1 FLOAT NOT NULL,
            UNIQUE (model, prompt_type, technique, evaluation_scope) -- MODIFIED
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



def calculate_f1(tp: int, fp: int, tn: int, fn: int) -> float:
    if (tp + fp) == 0: precision = 0.0
    else: precision = tp / (tp + fp)
    if (tp + fn) == 0: recall = 0.0
    else: recall = tp / (tp + fn)
    if (precision + recall) == 0: f1_score = 0.0
    else: f1_score = 2 * ((precision * recall) / (precision + recall))
    return f1_score

def macro_f1_formula(f1_scores: list) -> float:
    if not f1_scores: return 0.0
    return sum(f1_scores) / len(f1_scores)

def micro_f1_formula(f1_per_class: list) -> float:
    total_tp = sum(item['matrix_dict']['tp'] for item in f1_per_class)
    total_fp = sum(item['matrix_dict']['fp'] for item in f1_per_class)
    total_fn = sum(item['matrix_dict']['fn'] for item in f1_per_class)
    return calculate_f1(total_tp, total_fp, 0, total_fn) # TN is not used in F1

def calculate_failure_rate(predictions_df: pd.DataFrame) -> float:
    if predictions_df.empty: return 0.0
    total_preds = len(predictions_df)
    null_preds = predictions_df['prediction'].isnull().sum()
    return round(((null_preds / total_preds) * 100), 2)


def calculate_strict_f1(predictions_df: pd.DataFrame, label: str, con: db.DuckDBPyConnection) -> tuple:
    # This function is simplified assuming an 'agreed_label' column exists
    # in a table that can be joined.
    # For this example, we assume the logic is correct as provided.
    # A robust implementation would join with annotations and filter for agreement.
    label_pred_ids = predictions_df[predictions_df['prediction'] == label].id.tolist()
    neg_label_pred_ids = predictions_df[(predictions_df['prediction'] != label) & (predictions_df['prediction'].notna())].id.tolist()
    if not label_pred_ids: label_pred_ids = [None] # Handle empty lists for SQL IN clause
    if not neg_label_pred_ids: neg_label_pred_ids = [None]
    
    tp = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label = ?", (tuple(label_pred_ids), label)).fetchone()[0]
    fp = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label <> ?", (tuple(label_pred_ids), label)).fetchone()[0]
    tn = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label <> ?", (tuple(neg_label_pred_ids), label)).fetchone()[0]
    fn = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label = ?", (tuple(neg_label_pred_ids), label)).fetchone()[0]
    
    f1_score = calculate_f1(tp, fp, tn, fn)
    return (f1_score, {"tp": tp, "fp": fp, "tn": tn, "fn": fn})

def calculate_loose_f1(predictions_df: pd.DataFrame, label: str, con: db.DuckDBPyConnection) -> tuple:
    # This function is now self-contained and doesn't need a separate SQL query
    # as it operates on the passed DataFrame.
    # To make it work, we must first join predictions with annotations.
    # We'll do this in the main loop instead.
    is_positive_prediction = (predictions_df['prediction'] == label)
    is_positive_ground_truth = (predictions_df['maris_stance'] == label) | (predictions_df['harriet_stance'] == label)

    tp = (is_positive_prediction & is_positive_ground_truth).sum()
    fp = (is_positive_prediction & ~is_positive_ground_truth).sum()
    fn = (~is_positive_prediction & is_positive_ground_truth).sum()
    tn = (~is_positive_prediction & ~is_positive_ground_truth).sum()
    
    f1_score = calculate_f1(tp, fp, tn, fn)
    return (f1_score, {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})


def get_config_id(model: str, prompt_type: str, technique: str, evaluation_scope: str, con: db.DuckDBPyConnection) -> int:
    sql = "SELECT config_id FROM model_evaluation WHERE model = ? AND prompt_type = ? AND technique = ? AND evaluation_scope = ?"
    result = con.execute(sql, (model, prompt_type, technique, evaluation_scope)).fetchone()
    if result:
        return result[0]
    raise ValueError("Could not find config_id. Make sure the main record was inserted first.")

def insert_f1_scores(
    model: str, prompt_type: str, technique: str, evaluation_scope: str,
    failure_rate: float,
    strict_macro_f1: float, strict_micro_f1: float,
    loose_macro_f1: float, loose_micro_f1: float,
    con: db.DuckDBPyConnection
):
    insert_sql = """
        INSERT INTO model_evaluation 
            (model, prompt_type, technique, evaluation_scope, 
            failure_rate, strict_macro_f1, strict_micro_f1, 
            loose_macro_f1, loose_micro_f1) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (model, prompt_type, technique, evaluation_scope) DO NOTHING;
    """
    con.execute(insert_sql, (
        model, prompt_type, technique, evaluation_scope,
        failure_rate, strict_macro_f1, strict_micro_f1,
        loose_macro_f1, loose_micro_f1
    ))

def insert_label_and_matrix_data(
    config_id: int,
    strict_f1_per_class: list,
    loose_f1_per_class: list,
    con: db.DuckDBPyConnection
):
    label_records = []
    matrix_records = []

    for i in range(len(strict_f1_per_class)):
        strict_data = strict_f1_per_class[i]
        loose_data = loose_f1_per_class[i]
        label = strict_data['label']

        # Prepare label_f1 records
        label_records.append((config_id, label, loose_data['loose_f1_score'], strict_data['strict_f1_score']))

        # Prepare eval_matrix records
        s_matrix = strict_data['matrix_dict']
        l_matrix = loose_data['matrix_dict']
        matrix_records.append((config_id, label, 'strict', s_matrix['tp'], s_matrix['fp'], s_matrix['tn'], s_matrix['fn']))
        matrix_records.append((config_id, label, 'loose', l_matrix['tp'], l_matrix['fp'], l_matrix['tn'], l_matrix['fn']))

    label_sql = "INSERT INTO label_f1 (config_id, label, loose_f1, strict_f1) VALUES (?, ?, ?, ?)"
    matrix_sql = "INSERT INTO eval_matrix (config_id, label, eval_type, tp, fp, tn, fn) VALUES (?, ?, ?, ?, ?, ?, ?)"
    
    con.executemany(label_sql, label_records)
    con.executemany(matrix_sql, matrix_records)

def run_f1_calculations_for_subset(
    predictions_df: pd.DataFrame, 
    evaluation_scope: str, 
    con: db.DuckDBPyConnection
):
    """
    A generic function to calculate all F1 scores for a given DataFrame
    and insert them into the database with the correct scope.
    """
    if predictions_df.empty:
        print(f"Skipping {evaluation_scope} calculations: DataFrame is empty.")
        return

    # Extract metadata from the DataFrame
    model = predictions_df['model'].iloc[0]
    prompt_type = predictions_df['prompt_type'].iloc[0]
    technique = predictions_df['technique'].iloc[0]

    # The 'loose' calculation needs annotator stances, so we join them here.
    # This ensures the dataframe passed to calculate_loose_f1 has the required columns.
    join_sql = """
        SELECT p.*, a.maris_stance, a.harriet_stance
        FROM predictions_df p
        LEFT JOIN (
            SELECT maris.annotated_paragraph_id AS id, maris.stance AS maris_stance, harriet.stance AS harriet_stance
            FROM annotations AS maris JOIN annotations AS harriet
            ON maris.annotated_paragraph_id = harriet.annotated_paragraph_id
            WHERE maris.annotator = 'maris' AND harriet.annotator = 'harriet'
        ) a ON p.id = a.id
    """
    eval_df = con.execute(join_sql).fetchdf()


    labels = ['favour', 'against', 'neither']
    strict_f1_per_class = []
    loose_f1_per_class = []

    for label in labels:
        strict_f1_score, strict_matrix = calculate_strict_f1(eval_df, label, con)
        loose_f1_score, loose_matrix = calculate_loose_f1(eval_df, label, con)
        
        strict_f1_per_class.append({"label": label, "strict_f1_score": strict_f1_score, "matrix_dict": strict_matrix})
        loose_f1_per_class.append({"label": label, "loose_f1_score": loose_f1_score, "matrix_dict": loose_matrix})

    # Calculate macro and micro scores
    strict_macro_f1 = macro_f1_formula([d['strict_f1_score'] for d in strict_f1_per_class])
    strict_micro_f1 = micro_f1_formula(strict_f1_per_class)
    loose_macro_f1 = macro_f1_formula([d['loose_f1_score'] for d in loose_f1_per_class])
    loose_micro_f1 = micro_f1_formula(loose_f1_per_class)
    failure_rate = calculate_failure_rate(eval_df)
    
    # Use a transaction for insertions
    con.begin()
    # Insert the main evaluation record
    insert_f1_scores(
        model, prompt_type, technique, evaluation_scope, failure_rate,
        strict_macro_f1, strict_micro_f1, loose_macro_f1, loose_micro_f1, con
    )
    # Get the new config_id and insert the detailed records
    config_id = get_config_id(model, prompt_type, technique, evaluation_scope, con)
    insert_label_and_matrix_data(config_id, strict_f1_per_class, loose_f1_per_class, con)
    con.commit()
    print(f"Successfully evaluated and stored results for {model}/{technique}/{prompt_type} with scope: {evaluation_scope}")


def evaluate_test_predictions(con: db.DuckDBPyConnection):
    """ Evaluates all model / technique / prompt type combos present in the predictions table (test_data set).
    """
    models = [row[0] for row in con.execute("SELECT DISTINCT model FROM predictions;").fetchall()]
    techniques = [row[0] for row in con.execute("SELECT DISTINCT technique FROM predictions;").fetchall()]

    for model in models:
        for technique in techniques:
            prompt_types = [row[0] for row in con.execute("SELECT DISTINCT prompt_type FROM predictions WHERE model = ? AND technique = ?;", (model, technique)).fetchall()]
            for prompt_type in prompt_types:
                
                # --- 1. Evaluate on the OVERALL dataset ---
                overall_pred_df = con.execute(
                    "SELECT * FROM predictions WHERE model = ? AND technique = ? AND prompt_type = ?;",
                    (model, technique, prompt_type)
                ).fetchdf()
                run_f1_calculations_for_subset(overall_pred_df, 'overall', con)

                # --- 2. Evaluate on the AGREED_ONLY subset ---
                agreed_pred_sql = """
                    WITH LabelersAnnotations AS (
                        SELECT maris.annotated_paragraph_id, maris.stance AS maris_stance, harriet.stance AS harriet_stance
                        FROM annotations AS maris JOIN annotations AS harriet 
                        ON maris.annotated_paragraph_id = harriet.annotated_paragraph_id
                        WHERE maris.annotator = 'maris' AND harriet.annotator = 'harriet'
                    )
                    SELECT p.* FROM predictions AS p
                    JOIN LabelersAnnotations AS la ON p.id = la.annotated_paragraph_id
                    WHERE p.model = ? AND p.technique = ? AND p.prompt_type = ?
                    AND la.maris_stance = la.harriet_stance;
                """
                agreed_pred_df = con.execute(agreed_pred_sql, (model, technique, prompt_type)).fetchdf()
                run_f1_calculations_for_subset(agreed_pred_df, 'agreed_only', con)

def evaluate_engineering_predictions(con: db.DuckDBPyConnection):
    """ Evaluates all model / technique / prompt type combos present in the engineering_predictions table.
    """
    models = [row[0] for row in con.execute("SELECT DISTINCT model FROM engineering_predictions;").fetchall()]
    techniques = [row[0] for row in con.execute("SELECT DISTINCT technique FROM engineering_predictions;").fetchall()]

    for model in models:
        for technique in techniques:
            prompt_types = [row[0] for row in con.execute("SELECT DISTINCT prompt_type FROM engineering_predictions WHERE model = ? AND technique = ?;", (model, technique)).fetchall()]
            for prompt_type in prompt_types:
                
                # --- 1. Evaluate on the OVERALL dataset ---
                overall_pred_df = con.execute(
                    "SELECT * FROM engineering_predictions WHERE model = ? AND technique = ? AND prompt_type = ?;",
                    (model, technique, prompt_type)
                ).fetchdf()
                run_f1_calculations_for_subset(overall_pred_df, 'overall', con)

                # --- 2. Evaluate on the AGREED_ONLY subset ---
                agreed_pred_sql = """
                    WITH LabelersAnnotations AS (
                        SELECT maris.annotated_paragraph_id, maris.stance AS maris_stance, harriet.stance AS harriet_stance
                        FROM annotations AS maris JOIN annotations AS harriet 
                        ON maris.annotated_paragraph_id = harriet.annotated_paragraph_id
                        WHERE maris.annotator = 'maris' AND harriet.annotator = 'harriet'
                    )
                    SELECT p.* FROM engineering_predictions AS p
                    JOIN LabelersAnnotations AS la ON p.id = la.annotated_paragraph_id
                    WHERE p.model = ? AND p.technique = ? AND p.prompt_type = ?
                    AND la.maris_stance = la.harriet_stance;
                """
                agreed_pred_df = con.execute(agreed_pred_sql, (model, technique, prompt_type)).fetchdf()
                run_f1_calculations_for_subset(agreed_pred_df, 'agreed_only', con)
    
                
                
                


def main():
    con = connect_to_db()
    create_evaluation_table(con)
    
    parser = argparse.ArgumentParser(description="Evaluate model predictions from different datasets.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['test', 'engineering'],
        help="The dataset to evaluate: 'test' for the main predictions or 'engineering' for the engineering set."
    )
    args = parser.parse_args()

    # Determine which table to use based on the argument
    if args.dataset == 'test':
        evaluate_test_predictions(con)
    elif args.dataset == 'engineering': # args.dataset == 'engineering'
        evaluate_engineering_predictions(con)
    else:
        print("Error: Please provide an identifier, whether the test or engineering predictions should be evalutated (--dataset=engineering or --dataset=test)")
    
    con.close()



if __name__ == "__main__":
    main()