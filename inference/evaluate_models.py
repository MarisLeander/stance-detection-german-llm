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

def calculate_f1(tp:int, fp:int, tn:int, fn:int) -> int:
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
    total_sum = sum(item[1] for item in f1_scores)
    
    # Divides the sum by the total number of items.
    macro_f1 = total_sum / len(f1_scores)
    return macro_f1

def calculate_strict_f1(predictions_df:pd.DataFrame, label:str, con:db.DuckDBPyConnection) -> int:
    # Get ids of all predictions for current label
    label_pred_ids = predictions_df[predictions_df['prediction'] == label].id.tolist()
    # Get ids of all prediction for the other labels
    neg_label_pred_ids = predictions_df[predictions_df['prediction'] != label].id.tolist()
    # Get true pos, false pos, etc.
    tp = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label = ?", (tuple(label_pred_ids), label)).fetchone()[0]
    fp = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label <> ?", (tuple(label_pred_ids), label)).fetchone()[0]
    tn = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label <> ?", (tuple(neg_label_pred_ids), label)).fetchone()[0]
    fn = con.execute(f"SELECT count(*) FROM annotated_paragraphs WHERE id IN ? AND agreed_label = ?", (tuple(neg_label_pred_ids), label)).fetchone()[0]
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    f1_score = calculate_f1(tp, fp, tn, fn)
    print(f"F1-score: {f1_score}")
    return f1_score

def calculate_loose_f1(predictions_df:pd.DataFrame, label:str, con:db.DuckDBPyConnection) -> int:
    # Get ids of all predictions for current label
    label_pred_ids = predictions_df[predictions_df['prediction'] == label].id.tolist()
    # Get ids of all prediction for the other labels
    neg_label_pred_ids = predictions_df[predictions_df['prediction'] != label].id.tolist()
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
    )
    """
    con.execute(view_sql)
    
    # Either one of the annotators stance annotations has to agree on the predictions of the model
    tp_sql = """
        SELECT COUNT(DISTINCT annotated_paragraph_id)
        FROM 
            LabelersAnnotations
        WHERE 
            annotated_paragraph_id IN ? 
            AND (harriet_stance = ? OR maris_stance = ?);
    """
    tp = con.execute(tp_sql, (tuple(label_pred_ids), label, label)).fetchone()[0]
    # No one of the annotators stance annotations agrees on the predictions of the model
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

    #  Ground trouth is the label, i.e. one of the annotators annotated it, but the model annotated something else
    fn_sql = """
        SELECT COUNT(DISTINCT annotated_paragraph_id)
        FROM 
            LabelersAnnotations
        WHERE 
            annotated_paragraph_id IN ? 
            AND (harriet_stance = ? OR maris_stance = ?);
    """
    fn = con.execute(fn_sql, (tuple(neg_label_pred_ids), label, label)).fetchone()[0]
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    f1_score = calculate_f1(tp, fp, tn, fn)
    print(f"F1-score: {f1_score}")
    # Drop our view
    return f1_score

def calculate_macro_f1(predictions_df:pd.DataFrame, con:db.DuckDBPyConnection):
    labels = ['favour', 'against', 'neither']
    strict_f1_per_class = []
    loose_f1_per_class = []
    for label in labels:
        print(f"\nLabel: {label}")
        # For the strict f1 score the model has to predict the agreed on label, which is neither if the annotators labelled different labels.
        print("Calculating strict f1-score")
        strict_f1_score = calculate_strict_f1(predictions_df, label, con)
        strict_f1_per_class.append((label, strict_f1_score))
        # For the loose f1 score, the model has only to predict one of the labels each annotator labelled.
        print("\nCalculating loose f1-score")
        loose_f1_score = calculate_loose_f1(predictions_df, label, con)
        loose_f1_per_class.append((label, loose_f1_score))

    strict_macro_f1 = macro_f1_formula(strict_f1_per_class)
    print(f"Strict macro f1 = {strict_macro_f1}")
    loose_macro_f1 = macro_f1_formula(loose_f1_per_class)
    print(f"Loose macro f1 = {loose_macro_f1}")

def evaluate_predictions(con:db.DuckDBPyConnection):
    models = con.execute("SELECT DISTINCT model FROM predictions;").fetchall() 
    technique = con.execute("SELECT DISTINCT technique FROM predictions;").fetchall()
    for model in models:
        for technique in technique: 
            runs = con.execute("SELECT DISTINCT run FROM predictions WHERE model = ? AND technique = ?", (model[0], technique[0])).fetchall()
            for run in runs:
                print(f"Model: {model[0]} | Technique: {technique[0]} | Run: {run[0]}")
                predictions_df = con.execute("SELECT id, prediction FROM predictions WHERE model = ? AND technique = ? AND run = ?;", (model[0], technique[0], run[0])).df()
                calculate_macro_f1(predictions_df, con)
                


def main():
    con = connect_to_db()
    evaluate_predictions(con)



if __name__ == "__main__":
    main()