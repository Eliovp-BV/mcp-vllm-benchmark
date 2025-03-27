# server.py
from mcp.server.fastmcp import FastMCP
from typing import Dict
import concurrent.futures
import os
import requests
import pathlib
import tqdm

from benchmark_tool import run_benchmark

# Create an MCP server
mcp = FastMCP("vLLM Bencher")

@mcp.tool()
def benchmark_vllm(
    model: str,
    base_url: str,
    num_prompts: int = 10,
) -> Dict:
    """
    Run vLLM benchmarking tool to measure model performance
    
    Args:
        model: The model to benchmark (e.g., 'meta-llama/Llama-2-7b-hf')
        backend: Backend server to use (vllm, tgi, openai, etc.)
        dataset: Dataset to use for benchmarking (sharegpt, random, etc.)
        dataset_path: Path to the dataset file
        num_prompts: Number of prompts to benchmark with
        request_rate: Requests per second
        concurrent_requests: Number of concurrent requests
        max_tokens: Maximum number of tokens to generate
        vllm_dir: Directory where vLLM is installed
        api_url: URL of the API to benchmark
        save_result: Whether to save benchmark results
        result_filename: Filename to save benchmark results
        api_key: API key for the backend
        trust_remote_code: Whether to trust remote code
        extra_args: Additional arguments to pass to benchmark_serving.py
    
    Returns:
        Dictionary containing benchmark results including throughput, latency, and other metrics
    """
    
    # Define the dataset path
    dataset_filename = "ShareGPT_V3_unfiltered_cleaned_split.json"
    current_dir = pathlib.Path(__file__).parent.absolute()
    dataset_path = current_dir / dataset_filename
    
    # Check if dataset exists, if not, download it
    if not dataset_path.exists():
        dataset_url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        try:
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()
            
            # Get file size if available
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading dataset")
            
            with open(dataset_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            
            progress_bar.close()
        except Exception as e:
            # If download failed and partial file exists, remove it
            if dataset_path.exists():
                os.remove(dataset_path)
            raise

    # Run the benchmark in a separate thread to avoid asyncio event loop issues
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            run_benchmark,
            model=model,
            backend="vllm",
            dataset="sharegpt",
            dataset_path=str(dataset_path),
            num_prompts=num_prompts,
            base_url=base_url,
        )
        return future.result()

if __name__ == "__main__":
    mcp.run()
