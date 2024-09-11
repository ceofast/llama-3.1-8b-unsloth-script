# llama-3.1-8b-unsloth-script

# Language Model Training and Inference Project

This project contains scripts for fine-tuning a language model and setting up an inference API using FastAPI. It's designed to work with models from the Hugging Face ecosystem and utilizes the Unsloth library for optimization.

## Content

1. `main.py`:Script for fine-tuning a language model and pushing it to Huggingface Hub.
2. `app.py`: FastAPI application for serving the fine-tuned model.
3. `request.py`: Example script for making requests to the FastAPI server.

## Requirements

Will be added later

## Setup

1. Install the required dependencies:

bash`pip install torch transformers fastapi uvicorn unsloth trl datasets huggingface_hub

2. Additional dependencies may be installed by the scripts as neeeded.

## Usage 

### Fine-tuning the Model (main.py)

1. Run the script with the following command:

   bash```python3 main.py --model_name "unsloth/Meta-Llama-3.1.8B" --max_seq_length 2048 --load_in_4bit --oytput_dir "outputs" --hf_token "YOUR_HF_TOKEN" --repo_name "YOUR_REPO_NAME"```

2. The script will fine-tune the model on the Alpaca dataset and push the results to your Hugging Face repository.

## Stating the Inference Server (app.py)

1. Ensure that the fine-tuned model is available in the specified path.
2. Start the FastAPI server:

bash```python3 app.py```

3 The server will start on ```http://localhost:8000```

## Making Requests (request.py)

1. With the server running, you can use the `reuqest.py`script to make inference requests:

bash```python3 request.py```

2. Modify the `instructinon`and `input`in `request.py`to test different prompts.

## Configuration

* In `main.py`, you can adjust training parameters such as batch size, learning rate, and number of steps in the `TrainingArguments`section.
* In `app.py`, you can modify the model loading parameters and generation settings as needed.


# Notes

* Ensure you have sufficient GPU memory for training and inference, especially when using larger models.
* The `load_in_4bit` option in `main.py` enables 4-bit quantization for reduced memory usage.
* Remember to keep your Hugging Face token secure and do not share it publicly.

Contribution

Feel free open issues or submit pull requests for any improvements or bug fixes.
