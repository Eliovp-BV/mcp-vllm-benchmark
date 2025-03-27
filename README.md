# MCP vLLM Benchmarking Tool

This is proof of concept on how to use MCP to interactively benchmark vLLM.

## Usage

1. Clone the repository
2. Add it to your MCP servers:
```
{
    "mcpServers": {
        "mcp-vllm": {
            "command": "uv",
            "args": [
                "run",
                "/Path/TO/mcp-vllm-benchmarking-tool/server.py"
            ]
        }
    }
}
```

Then you can prompt for example like this:

```
Do a vllm benchmark for this endpoint: http://10.0.101.39:8888 
benchmark the following model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B 
run the benchmark 3 times with each 32 num prompts, then compare the results, but ignore the first iteration as that is just a warmup.
```


## Todo:

- Due to some random outputs by vllm it may show that it found some invalid json. I have not really looked into it yet.