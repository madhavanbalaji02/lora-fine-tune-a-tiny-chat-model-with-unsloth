"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,          # Automatically detects your hardware (e.g., Float16/Bfloat16)
        load_in_4bit=True,   # Loads the weights in 4-bit quantization to fit on smaller GPUs
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    try:
        import bitsandbytes as bnb
        target_class = bnb.nn.Linear4bit
    except ImportError:
        # Fallback if bitsandbytes isn't fully installed or imported in the test env,
        # checking the class name directly is a robust alternative.
        target_class = None

    for submodule in model.modules():
        if target_class and isinstance(submodule, target_class):
            return True
        if submodule.__class__.__name__ == "Linear4bit":
            return True
            
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    from unsloth import FastLanguageModel
    
    if target_modules is None:
        target_modules = get_lora_target_modules()
        
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return float(trainable_count) / float(total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    return [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Explain photosynthesis in one sentence.",
            "response": "Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        },
        {
            "instruction": "Convert 100 degrees Celsius to Fahrenheit.",
            "response": "100 degrees Celsius is equal to 212 degrees Fahrenheit."
        }
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)["input_ids"]

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    # TODO: build TrainingArguments with batch size 1, given max_steps, given lr, bf16 or fp16.
    import torch
    from transformers import TrainingArguments

    # Determine optimal precision format based on GPU support
    use_bf16 = torch.cuda.is_bf16_supported()

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=1,
        optim="adamw_8bit",  # Keeps memory consumption low
        seed=3407,           # Unsloth stable seed recommendation
    )

# Step 16 - build_sft_trainer
def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    # TODO: wire model, tokenizer, dataset, and training_args into an SFTTrainer
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # Explicitly disable sequence packing as requested
        args=training_args,
    )
    return trainer

# Step 17 - run_sft_training
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    import sys
    
    # Resolve the sandbox module-reloading conflict by syncing SFTConfig references
    if hasattr(trainer, "args"):
        config_class = type(trainer.args)
        if config_class.__name__ == "SFTConfig":
            if "trl.trainer.sft_config" in sys.modules:
                sys.modules["trl.trainer.sft_config"].SFTConfig = config_class
            if "trl" in sys.modules and hasattr(sys.modules["trl"], "SFTConfig"):
                sys.modules["trl"].SFTConfig = config_class

    # Ensure single-process execution to eliminate multiprocessing pickling entirely
    if hasattr(trainer, "dataset_num_proc"):
        trainer.dataset_num_proc = None
    if hasattr(trainer, "args"):
        trainer.args.dataloader_num_workers = 0
        if hasattr(trainer.args, "dataset_num_proc"):
            trainer.args.dataset_num_proc = None

    # Execute the training steps
    train_result = trainer.train()
    
    # Extract total training loss from the execution metrics
    final_loss = train_result.metrics.get("train_loss", 0.0)
    return float(final_loss)

# Step 18 - switch_to_inference_mode
def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    # TODO: call the Unsloth helper that prepares the model for fast generation
    from unsloth import FastLanguageModel
    
    FastLanguageModel.for_inference(model)
    return model

# Step 19 - build_chat_prompt
def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    # TODO: wrap the instruction as a user turn and produce the assistant-generation prompt string
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Step 20 - generate_reply
def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    # TODO: tokenize prompt, run model.generate with do_sample=False, decode new tokens only
    # Tokenize the input prompt text and move tensors to the model's current device
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Generate tokens deterministically using greedy decoding
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=False,
        use_cache=True
    )
    
    # Isolate only the newly generated tokens by slicing past the prompt length
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    
    # Decode the newly generated token ids back into a text string
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

