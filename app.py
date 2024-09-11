from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

app = FastAPI()

# Model ve tokenizer'ı global olarak yükleyin
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # Eğitilmiş modelinizin yolu
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # 2x daha hızlı çıkarım için

class Query(BaseModel):
    instruction: str
    input: str = ""

def format_prompt(instruction: str, input_text: str = "") -> str:
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

@app.post("/generate")
async def generate_text(query: Query):
    prompt = format_prompt(query.instruction, query.input)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    try:
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
