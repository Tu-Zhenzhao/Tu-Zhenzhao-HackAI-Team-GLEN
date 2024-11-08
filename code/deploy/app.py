from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

app = FastAPI()

# Load model and tokenizer once during startup
base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
new_model_path = "/project/models/NV-llama3.1-8b-Arxiv"
api_key = ""

special_tokens = {
    'bos_token': "<bos>",
    'eos_token': "<eos>",
    'pad_token': "<pad>",
    'additional_special_tokens': ["<user>", "</user>", "<assistant>", "</assistant>"]
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=api_key)
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = special_tokens['pad_token']

# Load the model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    token=api_key,
    quantization_config=bnb_config,
    cache_dir="/project/models",
    device_map="auto"
)
base_model.resize_token_embeddings(len(tokenizer))

# Load the fine-tuned LoRA model
peft_model = PeftModel.from_pretrained(base_model, "/project/models/arxiv_model").to("cuda")

# Request model
class InferenceRequest(BaseModel):
    instruction: str

# Define the format_example function
def format_example(instruction, response=""):
    return f"<bos>\n<user>\n{instruction}\n</user>\n<assistant>\n{response}"

@app.post("/generate/")
async def generate_text(request: InferenceRequest):
    input_text = format_example(request.instruction)
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to("cuda")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate response
    with torch.no_grad():
        output = peft_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5,
            num_return_sequences=1,
            eos_token_id=tokenizer.convert_tokens_to_ids("<eos>"),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>")
        )

    # Decode and return response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)