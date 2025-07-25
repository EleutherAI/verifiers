import verifiers as vf
from verifiers.tools import python
from verifiers.utils.data_utils import load_example_dataset
import argparse
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

"""
Math Python Example - Train or Evaluate

TRAINING:
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'willcb/Qwen2.5-7B-Math-Python-SFT' --tensor-parallel-size 4

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_python.py --mode train

EVALUATION:
Uses OpenRouter API (set OPENROUTER_API_KEY in .env file)

python verifiers/examples/math_python.py --mode eval
"""

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\nx = sympy.symbols('x')\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

After concluding your message with a tool call,
you will then see the tool's output inside <result> tags as a new message. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible, rather than using your own knowledge.

The <answer>...</answer> tags should contain only your final answer as a numeric expression.

Example:
<think>
Let's submit the answer.
</think>
<answer>
\\frac{{1}}{{2}}
</answer>
"""

dataset = load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    max_steps=3,
    sampling_args={"extra_body": {"logprobs": True}}
)
print(vf_env.system_prompt)

def train():
    """Train the model using GRPO"""
    model_name = "willcb/Qwen2.5-7B-Math-Python-SFT"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    run_name = "math-grpo_" + model_name.split("/")[-1].lower()

    training_args=vf.grpo_defaults(run_name=run_name)
    training_args.num_iterations=2
    training_args.per_device_train_batch_size=8
    training_args.num_generations=8
    training_args.gradient_accumulation_steps=2

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    trainer.train()

def evaluate():
    """Evaluate using OpenRouter"""
    print("Evaluating with OpenRouter...")
    
    # Create OpenRouter client
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/EleutherAI/verifiers",
            "X-Title": "Verifiers Math Python Evaluation",
        }
    )
    
    # Evaluate on a subset
    results = vf_env.evaluate(
        client=client,
        model="moonshotai/kimi-k2",  # Good at math and code
        num_examples=10,  # Test on 10 examples
        score_rollouts=True
    )
    
    print(f"Results:")
    print(f"Average reward: {results.get('reward', results.get('rewards_avg', 'N/A'))}")
    if 'reward' in results and isinstance(results['reward'], list):
        success_rate = sum(1 for r in results['reward'] if r > 0) / len(results['reward'])
        print(f"Success rate: {success_rate:.2%}")
    elif 'rewards' in results:
        success_rate = sum(1 for r in results['rewards'] if r > 0) / len(results['rewards'])
        print(f"Success rate: {success_rate:.2%}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math Python Example - Train or Evaluate")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", 
                       help="Mode to run: train or eval")
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate()