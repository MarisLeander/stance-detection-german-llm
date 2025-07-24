# Appeal, Align, Divide? Stance Detection in German Political Texts

This repository contains the code and resources for my Bachelor's thesis.

-   **Title:** *Appeal, Align, Divide? Stance Detection on Group-Directed Messaging in German Political Texts Using Large Language Models*
-   **Institution:** Data and Web Science Group, Prof. Dr. Simone Paolo Ponzetto, University of Mannheim
-   **Supplementary Material:** The pre-built database and other materials can be found [here](https://drive.google.com/drive/folders/1ZuMQNow-ZOQVzNSS2GgwxcOnEz5UNhOq?usp=sharing).

---

## 📋 Requirements

### Hardware
-   To run the local LLM (`gemma-3-27b-it`), a high-performance GPU is required. The minimum requirement is an **NVIDIA H100 NVL GPU with 94 GB of VRAM**.

### Configuration
1.  **API Keys:** Create a `secrets.json` file in the project's root directory. It must contain your API keys in the following format:
    ```json
    {
        "gemini_api_key": "YOUR_GEMINI_API_KEY",
        "huggingface_api_key": "YOUR_HUGGINGFACE_API_KEY"
    }
    ```
2.  **Project Path:** The scripts use relative paths assuming the project folder (`stance-detection-german-llm`) is placed directly in your system's home directory (e.g., `~/stance-detection-german-llm`).

---

## 🚀 Getting Started

### 1. Database Setup
A database is required to run the classification scripts.

#### Recommended Method (Pre-built)
It is **highly recommended** to download the pre-built database from [here](https://drive.google.com/drive/folders/1cO_2MmCOKK2pqSWUgwWIKWYSs3Pux8RG?usp=sharing) to save time. This database already contains the classified group mentions.

#### Manual Method
Alternatively, you can build the database from scratch:
1.  Download the German parliamentary debates from [here](https://drive.google.com/drive/folders/1cO_2MmCOKK2pqSWUgwWIKWYSs3Pux8RG?usp=sharing).
2.  Run the Jupyter notebook `save_plenary_minutes.ipynb` to parse the debates and build the initial database.

### 2. Group Mention Classification
**Note:** This step is only necessary if you wish to re-classify the group mentions. The pre-built database already includes these classifications.

1.  Download the fine-tuned classifier (`bert-base-german-cased-finetuned-MOPE-L3_Run_3_Epochs_29`) from the official [MOPE repository](https://github.com/umanlp/mope).
2.  Create a `models/` folder in the project's root directory and place the downloaded classifier inside it.
3.  Run the extraction script. The `--reset_db` argument must be passed to clear existing data from the relevant tables.
    ```bash
    python extract-group-mention/extract_group_mention.py --reset_db
    ```

---

## ✍️ Annotation Data

This section outlines the process for extracting, inserting, and evaluating annotation data.

### Data Extraction
To extract a sample of data for annotation, run the following script:
```bash
python data-processing/extract_annotation_data.py
```

### Data Insertion
To insert manually annotated data back into the database:
1.  Place the annotators' completed files into the `/data/annotated_data/` folder.
2.  Run the processing script with the `--reset_db` argument. This will reset the corresponding tables before inserting the new data.
    ```bash
    python data-processing/process_annotated_data.py --reset_db
    ```

### Calculate Annotator Agreement
To calculate the inter-annotator agreement, run the script:
```bash
python data-processing/annotator_agreement.py
```

---

## 🤖 Running Stance Detection Models

- **Important:** Before running a new inference, you may need to manually delete previous predictions from the database (e.g., `DELETE FROM [engineering_predictions, predictions] WHERE [CONDITION]'`). This is the **recommended** way, to avoid deleting prior results.
- The whole database for predictions can be reset with running:
  ```bash
  inference/build_inference_table.py --reset_predictions
  ```
    
### Gemini-2.5-Pro

#### Engineering Dataset
To run inference on the engineering dataset, execute the script:
```bash
python inference/gemini_engineering_inference.py
```

#### Test Dataset
Inference on the test set is designed to be run in parallel for different configurations.

1.  **Run Inference:** Call the script from the CLI for each `prompt_type` and `technique` combination.
    ```bash
    python inference/gemini_inference.py --api-key=YOUR_GEMINI_API_KEY --prompt-type=it-thinking_guideline_higher_standards --technique=zero-shot
    ```
    *(Available prompt types can be found in `inference/inference_helper.py`)*

2.  **Insert Results:** After the script generates a CSV output file, insert the results into the database using a SQL client:
    ```sql
    INSERT INTO predictions (paragraph_id, model, prompt_type, technique, prediction, thoughts, thinking_process)
    SELECT paragraph_id, model, prompt_type, technique, prediction, thoughts, thinking_process 
    FROM read_csv('it-thinking_guideline_higher_standards_zero-shot.csv')
    ON CONFLICT DO NOTHING;
    ```

### Gemma-27b-it (Local Model)

To run inference for both the engineering and test datasets using the local Gemma model, execute the scripts:
```bash
python inference/gemma_engineering_inference.py
```

```bash
python inference/gemma_inference.py
```

---
## 📈 Evaluate Results

1. Run the following script with either `--dataset=test` or `--dataset=engineering`, to respectively evaluate the engineering or test dataset:
   ```bash
   python inference/evaluate_models.py --dataset=test
   ```
3. inference/evaluate_models.py
4. Query the corresponding database tables (model_evaluation, label_f1, eval_matrix using SQL:
   ```sql
   SELECT * FROM model_evaluation ORDER BY strict_macro_f1 DESC; --As an example
   ```
---

## 🔧 Customization & Known Issues

### Adding New Prompt Types
New prompt types for the models can be added by modifying the `inference/inference_helper.py` script.

### Known Issues
-   The `inference_helper.py` script is complex and could be improved. It is recommended to refactor it to dynamically parse prompts from a structured file (e.g., a JSON file) to better manage the different `prompt_type` and `technique` combinations.
