import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb as db
import numpy as np

def calculate_overall_kappa(con: db.DuckDBPyConnection):
    """ 
    Calculates Cohen's Kappa and generates a final, styled confusion matrix plot.
    
    The plot highlights all values within the main matrix in bold and includes
    marginal totals in a standard font weight. It is saved as 'confusion_matrix_final.pdf'.
    """
    # Get dataframe
    df = con.execute("SELECT * FROM annotations WHERE stance <> 'not a group';").fetchdf()
    
    # Reshape the data
    pivot_df = df.pivot(index='annotated_paragraph_id', columns='annotator', values='stance').reset_index()
    pivot_df.dropna(subset=['harriet', 'maris'], inplace=True)
    
    annotator_a_labels = pivot_df['harriet']
    annotator_b_labels = pivot_df['maris']
    
    # --- Calculate Cohen's Kappa ---
    kappa_score = cohen_kappa_score(annotator_a_labels, annotator_b_labels)
    print(f"Cohen's Kappa Score: {kappa_score:.4f}")

    # --- Generate and Save Confusion Matrix Plot with Totals ---
    
    # Define the order of labels
    labels = sorted(df['stance'].unique())
    
    # Calculate the confusion matrix
    cm = confusion_matrix(annotator_a_labels, annotator_b_labels, labels=labels)
    
    # Calculate row and column sums
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0)
    total_sum = cm.sum()

    # Create a new matrix with totals
    cm_with_row_totals = np.concatenate([cm, row_sums], axis=1)
    total_row = np.append(col_sums, total_sum)
    cm_with_totals = np.concatenate([cm_with_row_totals, [total_row]], axis=0)

    # Create new labels for the heatmap
    labels_with_total = labels + ['Total']
    
    # Create a heatmap plot
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    heatmap = sns.heatmap(
        cm_with_totals, 
        annot=True, 
        fmt='d', # Use standard integer formatting
        cmap='Blues',
        xticklabels=labels_with_total, 
        yticklabels=labels_with_total,
        linewidths=.5,
        annot_kws={"size": 14},
        cbar=True,
        cbar_kws={'label': 'Number of Agreements'}
    )
    

    # Add lines to separate the main matrix from the totals
    heatmap.axhline(y=len(labels), color='k', linewidth=2)
    heatmap.axvline(x=len(labels), color='k', linewidth=2)

    # Set titles and labels
    plt.title("Confusion Matrix for Annotator Agreement", fontsize=16, pad=20)
    plt.ylabel("Annotator: Harriet", fontsize=12)
    plt.xlabel("Annotator: Maris", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Save the plot to a PDF file
    plt.savefig('confusion_matrix_final.pdf', bbox_inches='tight')
    
    print("Final confusion matrix plot saved as 'confusion_matrix_final.pdf'")
    
    # To display the plot if running interactively:
    # plt.show()

def calculate_kappa_evolution(con):
    data = []
    for i in range(200, 3200, 200):
        sql = """
            SELECT * 
            FROM annotations a 
            JOIN annotated_paragraphs ap ON a.annotated_paragraph_id = ap.id 
            WHERE (stance <> 'not a group') AND ((LENGTH(ap.inference_paragraph) >= (? - 200)) AND (LENGTH(ap.inference_paragraph) < ?));
            """
        df = con.execute(sql, (i,i)).fetchdf()        
        # Reshape the data so each annotator has their own column.
        # The 'text_id' aligns the ratings for the same item.
        pivot_df = df.pivot(index='annotated_paragraph_id', columns='annotator', values='stance').reset_index()
        
        # These are the two sets of ratings that will be compared.
        annotator_a_labels = pivot_df['harriet']
        annotator_b_labels = pivot_df['maris']
        
        
        # calculate cohen's kappa
        kappa_score = cohen_kappa_score(annotator_a_labels, annotator_b_labels)
        # print(f"Paragraph length:  >={(i - 200)} and <{i}, Cohen's Kappa Score: {kappa_score:.4f}")
        data.append((i, round(kappa_score, 4), int(df.shape[0] / 2)))
    plot_evolution(data)

def plot_evolution(data):
    # --- 2. Prepare Data for Plotting ---
    # Create labels for the x-axis (e.g., "0-199", "200-399")
    x_labels = [f"{d[0]-200}-{d[0]-1}" for d in data]
    # Extract the kappa scores for the y-axis
    kappa_scores = [d[1] for d in data]
    # Extract the sample sizes
    sample_sizes = [d[2] for d in data]
    
    # --- 3. Create the Bar Chart ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(x_labels, kappa_scores, color='skyblue', edgecolor='black')
    
    # Add titles and labels
    ax.set_title("Cohen's Kappa Score by Paragraph Length (Amount of characters)", fontsize=16, pad=20)
    ax.set_xlabel("Paragraph Length Range", fontsize=12)
    ax.set_ylabel("Cohen's Kappa (κ)", fontsize=12)
    
    # Set y-axis limits for a consistent scale
    ax.set_ylim(0, 1.05) # Increased slightly to give text on top more space
    
    # Add a horizontal line for the "Substantial Agreement" threshold
    ax.axhline(y=0.61, color='r', linestyle='--', label='Overall agreement (0.61)')
    ax.legend()
    
    # --- 4. Add Labels to Each Bar (with conditional placement) ---
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        kappa_val = kappa_scores[i]
        n_val = sample_sizes[i]
        
        # Add the kappa score on top of the bar
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{kappa_val:.2f}', 
                ha='center', va='bottom', fontsize=11, color='black', weight='bold')
        
        # --- FIX: Conditional placement for the sample size (n) ---
        # If the bar is too short, place 'n' above the kappa score.
        # Otherwise, place it inside the bar.
        if yval < 0.1:
            # Place 'n' text above the kappa score text
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.06, f'n = {n_val}', 
                    ha='center', va='bottom', fontsize=9, color='black')
        else:
            # Place 'n' text inside the bar, near the top
            ax.text(bar.get_x() + bar.get_width()/2.0, yval - 0.03, f'n = {n_val}', 
                    ha='center', va='top', fontsize=9, color='black')

    
    # Rotate x-axis labels for better fit
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # --- 5. Save the Plot ---
    output_filename = 'kappa_by_length_barchart_with_n.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Bar chart with sample sizes saved successfully as '{output_filename}'")



def calculate_agreement_counts(con):
    """
    Calculates and prints the counts of agreed-upon labels for each paragraph length bin.
    """
    print("\n--- Agreed Label Counts per Length Bin ---")
    
    # Define the set of all possible labels to ensure consistent reporting
    all_possible_labels = {'favour', 'against', 'neither'}

    # Loop through the upper bound of each length bin
    for i in range(200, 3200, 200):
        lower_bound = i - 200
        sql = """
            SELECT * FROM annotations a 
            JOIN annotated_paragraphs ap ON a.annotated_paragraph_id = ap.id 
            WHERE (stance <> 'not a group') AND ((LENGTH(ap.inference_paragraph) >= ?) AND (LENGTH(ap.inference_paragraph) < ?));
            """
        df = con.execute(sql, (lower_bound, i)).fetchdf()
        
        if df.empty:
            print(f"Length < {i}: No annotations found.")
            continue

        # Reshape the data and keep only overlapping annotations
        pivot_df = df.pivot(index='annotated_paragraph_id', columns='annotator', values='stance')
        pivot_df.dropna(inplace=True)

        if pivot_df.empty:
            print(f"Length < {i}: No overlapping annotations.")
            continue
        
        # --- Core Logic to find and count agreements ---
        
        # 1. Filter the DataFrame to find rows where both annotators agreed
        agreed_df = pivot_df[pivot_df['harriet'] == pivot_df['maris']]
        
        # 2. Count the occurrences of each agreed-upon label
        agreement_counts = agreed_df['harriet'].value_counts().to_dict()

        # 3. Format the output string for printing
        # Ensure all possible labels are represented, even if their count is 0
        output_parts = []
        for label in sorted(list(all_possible_labels)):
            count = agreement_counts.get(label, 0)
            output_parts.append(f"{label} = {count}")
        
        output_str = ", ".join(output_parts)
        
        print(f"Length < {i}: {output_str} (Total Agreed: {len(agreed_df)})")

def plot_stance_distribution(con, annotator):
    """
    Fetches stance counts for paragraph length bins and creates a stacked bar chart.
    
    Args:
        con: An active DuckDB database connection.
    """
    trends_data = []
    all_stances = ['favour', 'against', 'neither'] # Define order for consistency

    # --- 1. Aggregate Data ---
    # Loop through each 200-character length bin
    for i in range(200, 3200, 200):
        lower_bound = i - 200
        sql = """
            SELECT
                a.stance
            FROM
                annotations AS a
            JOIN
                annotated_paragraphs AS ap 
                ON a.annotated_paragraph_id = ap.id
            WHERE
                a.annotator = ?
                AND a.stance <> 'not a group'
                AND LENGTH(ap.inference_paragraph) >= ?
                AND LENGTH(ap.inference_paragraph) < ?;
            """
        df = con.execute(sql, (annotator, lower_bound, i)).fetchdf()
        
        # Get counts for each stance in the current bin
        stance_counts = df['stance'].value_counts()
        
        # Store the results, ensuring all stances are present (even if count is 0)
        bin_data = {
            'bin': f"{lower_bound}-{i-1}",
            'favour': stance_counts.get('favour', 0),
            'against': stance_counts.get('against', 0),
            'neither': stance_counts.get('neither', 0)
        }
        trends_data.append(bin_data)

    # --- 2. Prepare DataFrame for Plotting ---
    # Convert the list of dictionaries to a DataFrame
    trends_df = pd.DataFrame(trends_data)
    # Set the bin label as the index for plotting
    trends_df.set_index('bin', inplace=True)
    # Ensure the columns are in a consistent order
    trends_df = trends_df[all_stances]
    
    print("Aggregated Stance Counts:")
    print(trends_df)

    # --- 3. Create and Save the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Use pandas' built-in plotting for convenience
    ax = trends_df.plot(
        kind='bar', 
        stacked=True, 
        figsize=(14, 8),
        color=['#4CAF50', '#F44336', '#9E9E9E'] # Green, Red, Grey
    )

    # Customize the plot
    ax.set_title('Distribution of Stance Labels by Paragraph Length', fontsize=16, pad=20)
    ax.set_xlabel('Paragraph Length Range (Characters)', fontsize=12)
    ax.set_ylabel('Number of Labeled Paragraphs', fontsize=12)
    ax.legend(title='Stance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot to a file
    output_filename = f'{annotator}_stance_distribution_trend.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nChart saved successfully as '{output_filename}'")



def main():
    home_dir = Path.home()
    db_path = home_dir / "stance-detection-german-llm" / "data" / "database" / "german-parliament.duckdb"
    con = db.connect(database=db_path, read_only=True)
    calculate_overall_kappa(con)
    calculate_kappa_evolution(con)
    calculate_agreement_counts(con)
    plot_stance_distribution(con, 'harriet')
    plot_stance_distribution(con, 'maris')
    con.close()

if __name__ == "__main__":
    main()