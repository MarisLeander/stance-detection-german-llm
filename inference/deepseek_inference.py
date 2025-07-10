import duckdb as db
import time
import vllm
import pandas as pd
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path
from inference_helper import get_formatted_prompt, get_prompt_list

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

def load_deepseek_model() -> tuple[vllm.entrypoints.llm.LLM, str]:
    start_time = time.time()
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    print("Initializing LLM and its tokenizer with vLLM...")
    
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95  # Use 95% of GPU memory
    )
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"LLM setup took {elapsed_time:.2f} min.")

    # sampling parameters for generation.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1000)
    print(type(sampling_params))
    return llm, sampling_params

def parse_r1_response(response:str) -> tuple[str,str]:
    """
    Parses the thinking process and answer using a regular expression.
    This is robust to variations in whitespace or structure.

    Returns:
        A tuple containing (thinking_process, final_answer).
        Returns (None, None) if the pattern doesn't match.
    """
    # This pattern looks for content between the tags, allowing for whitespace
    # re.DOTALL makes the '.' character match newlines as well
    pattern = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)
    
    match = pattern.search(response_text)
    
    if match:
        # group(1) is the content from the first (.*?)
        # group(2) is the content from the second (.*?)
        thinking = match.group(1).strip()
        answer = match.group(2).strip()
        return thinking, answer
    else:
        return None, None


def deepseek_inference(paragraph:str, group:str, engineering_id:int, prompt_type:str, llm:vllm.entrypoints.llm.LLM, sampling_params, con:db.DuckDBPyConnection) -> None:
    prompt = get_formatted_prompt(paragraph, group, prompt_type)
    
    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        # The 'prompt' here will be the long, formatted string with special tokens
        original_prompt_info = output.prompt
        generated_text = output.outputs[0].text
        thinking, answer = parse_r1_response(output.outputs[0].text)
        print(f"Thinking: {thinking}\n Answer: {answer}")
        

def process_test_set(con:db.DuckDBPyConnection, llm:vllm.entrypoints.llm.LLM, sampling_params):
    test_set = con.execute("SELECT * FROM engineering_data JOIN annotated_paragraphs USING(id)").fetchdf()
    prompt_types = get_prompt_list(cot=False, few_shot=False)
    for prompt_type in prompt_types:
        for _,row in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Doing inference on DeepSeek-R1-Distill-Llama-70B and prompt: {prompt_type}"):
            paragraph = row['inference_paragraph']
            engineering_id = row['id']
            group = row['group_text']

            deepseek_inference(paragraph, group, engineering_id, prompt_type, llm, sampling_params, con)

def main():
    con = connect_to_db()
    llm, sampling_params = load_deepseek_model()
    process_test_set(con, llm, sampling_params)

if __name__ == "__main__":
    main()

    

