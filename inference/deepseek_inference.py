import time
import vllm
import pandas as pd
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path
from inference_helper import get_formatted_prompt, get_prompt_list, get_engineering_data, insert_prediction, get_test_batch

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
    
    match = pattern.search(response)
    
    if match:
        # group(1) is the content from the first (.*?)
        # group(2) is the content from the second (.*?)
        thinking = match.group(1).strip()
        prediction = match.group(2).strip()
        return thinking, prediction
    else:
        return None, None


def deepseek_inference(prompt_batch, llm:vllm.entrypoints.llm.LLM, sampling_params) -> None:

    for prompt_dict in prompt_batch:
        prompt = prompt_dict.get("prompt")
        outputs = llm.generate([prompt], sampling_params)    
        for output in outputs:
            # The 'prompt' here will be the long, formatted string with special tokens
            original_prompt_info = output.prompt
            generated_text = output.outputs[0].text
            thinking, prediction = parse_r1_response(output.outputs[0].text)
            print(f"Thinking: {thinking}\n Answer: {prediction}")
            paragraph_id = prompt_dict.get("paragraph_id")
            prompt_type = prompt_dict.get("prompt_type")
            #insert_prediction(paragraph_id, 'DeepSeek-R1-Distill-Llama-70B', prompt_type, 'zero_shot', prediction, thinking, None)
        

def process_test_set(llm:vllm.entrypoints.llm.LLM, sampling_params):
    to_be_predicted_batch = get_engineering_data(sample_size=1)
    prompt_types = get_prompt_list(cot=False, few_shot=False)
    for prompt_type in prompt_types:
        prompt_batch = get_test_batch(to_be_predicted_batch, prompt_type)
        deepseek_inference(prompt_batch, llm, sampling_params)

def main():
    llm, sampling_params = load_deepseek_model()
    # llm, sampling_params = None, None
    process_test_set(llm, sampling_params)

if __name__ == "__main__":
    main()

    

