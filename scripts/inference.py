import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_agent():
    model_id = "focustiki/eigentiki"
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # We load in 4-bit to save memory if running locally
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True,
    )

    print("\nModel loaded. Preparing the test task...")
    
    messages = [
        {
            "role": "system", 
            "content": "You are a smart coding assistant. You have access to a tool named `run_python_code(code: str)`. You must use this tool if you need to run code."
        },
        {
            "role": "user", 
            "content": "Can you write a python script to calculate the square root of 144 and run it for me?"
        }
    ]

    # Use ChatML format since we trained with it
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\nAsking the agent to solve the task...")
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    print("\n--- Agent Response ---")
    # We print only the new generated text
    response_clean = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")
    print(response_clean.strip())
    print("----------------------")

if __name__ == "__main__":
    test_agent()
