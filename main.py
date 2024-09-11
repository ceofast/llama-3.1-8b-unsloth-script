import torch
from torch import __version__
import argparse
import subprocess
from trl import SFTTrainer
from datasets import load_dataset
from packaging.version import Version as V
from huggingface_hub import HfApi, create_repo
from transformers import TrainingArguments, TextStreamer
from unsloth import FastLanguageModel, is_bfloat16_supported


subprocess.run(["pip", "install", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
subprocess.run(["pip", "install", "--no-deps", xformers, "trl", "peft", "accelerate", "bitsandbytes", "triton", "huggingface_hub"], check=True)



def parser_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model and push to Hugging Face")
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B", help="Name of the pre-trained model")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saved model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--repo_name", type=str, required=True, help="Name for the Hugging Face repository")
    return parser.parse_args()

def format_prompt(instruction, input_text, output=""):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_text}

    ### Response:
    {output}"""

def formatting_prompts_func(examples):
    texts = [format_prompt(instr, inp, out) + tokenizer.eos_token for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
    return {"text": texts}

def train_model(args):
    global tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
        ),
    )

    trainer_stats = trainer.train()
    print_stats(trainer_stats)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Push to Hugging Face
    push_to_hub(args.output_dir, args.repo_name, args.hf_token)

def print_stats(trainer_stats):
    gpu_stats = torch.cuda.get_device_properties(0)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds")
    print(f"Peak reserved memory = {used_memory} GB")
    print(f"Peak reserved memory % of max memory = {round(used_memory/max_memory*100, 3)}%")

def push_to_hub(local_dir, repo_name, token):
    api = HfApi()
    repo_id = f"username/{repo_name}"
    api.create_repo(repo_id=repo_name, token=token, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_name,
        repo_type="model",
        token=token,
    )
    print(f"Model pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")

def main():
    args = parser_args()
    train_model(args)

if __name__ == "__main__":
    main()
