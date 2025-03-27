import argparse
from typing import Dict, List, Optional
from benchmark.benchmark_serving import main as main_benchmark_serving

def run_benchmark(
    model: str,
    base_url: str,
    backend: str = "vllm",
    dataset: str = "sharegpt",
    dataset_path: Optional[str] = None,
    num_prompts: int = 100,
    request_rate: float = 10.0,
    concurrent_requests: int = 10,
    max_tokens: int = 128,
    vllm_dir: Optional[str] = None,
    save_result: bool = True,
    result_filename: Optional[str] = None,
    api_key: Optional[str] = None,
    trust_remote_code: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Dict:
    """
    Run vLLM benchmarking tool
    
    Args:
        model: The model to benchmark
        backend: Backend server to use (vllm, tgi, openai, etc.)
        dataset: Dataset to use for benchmarking
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
        Dictionary with benchmark results
    """
    # Create argparse.Namespace object to pass to main_benchmark_serving
    args = argparse.Namespace()
    
    # Required parameters
    args.model = model
    args.backend = backend
    args.dataset_name = dataset
    args.dataset_path = dataset_path if dataset_path else "./ShareGPT_V3_unfiltered_cleaned_split.json"
    args.num_prompts = num_prompts
    args.request_rate = request_rate
    args.max_concurrency = concurrent_requests
    
    # Optional parameters with defaults
    args.host = "127.0.0.1"
    args.port = 8000
    args.endpoint = "/v1/completions"
    args.base_url = base_url
    args.seed = 0
    args.disable_tqdm = False
    args.profile = False
    args.use_beam_search = False
    args.tokenizer = None
    args.logprobs = None
    args.burstiness = 1.0
    args.ignore_eos = False
    args.percentile_metrics = "ttft,tpot,itl"
    args.metric_percentiles = "99"
    args.save_result = save_result
    args.save_detailed = False
    args.metadata = None
    args.result_dir = None
    args.result_filename = result_filename
    args.trust_remote_code = trust_remote_code
    args.tokenizer_mode = "auto"
    args.served_model_name = None
    args.lora_modules = None
    
    # Dataset-specific parameters
    args.sonnet_input_len = 550
    args.sonnet_output_len = 150
    args.sonnet_prefix_len = 200
    args.sharegpt_output_len = max_tokens
    args.random_input_len = 1024
    args.random_output_len = max_tokens
    args.random_range_ratio = 1.0
    args.random_prefix_len = 0
    args.hf_subset = None
    args.hf_split = None
    args.hf_output_len = max_tokens
    args.goodput = None
    
    # Handle extra args if provided
    if extra_args:
        for arg in extra_args:
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.lstrip('-').replace('-', '_')
                try:
                    # Try to convert to appropriate type
                    if value.lower() in ('true', 'yes'):
                        value = True
                    elif value.lower() in ('false', 'no'):
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                except (ValueError, AttributeError):
                    pass
                setattr(args, key, value)
    
    benchmark_result = main_benchmark_serving(args)
    return benchmark_result
