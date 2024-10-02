# Project Name
Research paper query engine for use with Wolfram Language.

## Description
We used a fine-tuning model to create a AI chatbot for querying arxiv research papers. We have automatically downloaded 80 academic papers. Within a month we learned how to use AI Workbench.

## Project Overview

This project demonstrates how to fine-tune a Large Language Model (LLM) using Low-Rank Adaptation (LoRA) for the purpose of recommending research papers based on user queries. The workflow focuses on efficiently handling large models by leveraging 4-bit precision techniques and LoRA to reduce memory consumption while retaining performance. The model takes user queries, matches them with topics from research paper titles, and generates contextually relevant responses.

### Features and Functionality

- Custom Dataset Handling: Loads a CSV of research papers (titles and summaries) and formats them for model training.
- Fine-Tuning with LoRA: Applies LoRA to specific layers of a pretrained LLM, optimizing for efficient fine-tuning.
- Query-Based Recommendations: The model takes user queries, processes them, and recommends research papers by generating an informative response.
- Tokenization with Special Tokens: Utilizes special tokens (<user>, <assistant>, etc.) to format input data for structured conversations.

## Instructions for Starting the Program and Asking Questions

### Target Systems

This project is designed to run on AI Workbench or systems with the following specifications:

- Operating System: Linux-based systems are recommended (Ubuntu 20.04+).
- GPU Support: Requires a CUDA-enabled GPU for training and inference (NVIDIA RTX 4090 24GB is used in this project).
- Python Version: Python 3.8 or higher.

### Hardware Requirements

- GPU: Minimum 24GB of VRAM is recommended for fine-tuning tasks. This project is optimized for systems with an NVIDIA RTX 4090 or higher.
- RAM: At least 16GB of system memory is recommended for handling the dataset and running the training efficiently.
- Disk Space: Ensure that you have at least 50GB of available disk space for model checkpoints and data processing.

### 1. **Environment Setup**:
   - Ensure you have Python 3.x installed along with the necessary libraries. You can install the required libraries using the following command:
     ```bash
     pip install torch transformers peft pandas
     ```
   - Make sure your system has access to a CUDA-enabled GPU, as this program requires GPU acceleration. Verify GPU availability by running:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```

### 2. **Running the Program**:
   - After cloning the project repository, navigate to the project directory in AI Workbench
   - Ensure you have the dataset (`ml_papers.csv`) placed in the correct directory as expected by the script.
   - Start the program by running the main notebook or script
   - Open the notebook `arxiv_model.ipynb` and execute the cells step by step.
### 3. **Asking Questions (Querying the Model)**:
   - In the notebook, locate the section where the chatbot is queried for paper recommendations.
   - Format your input as a user query. For example:
     ```python
     instruction = "I am looking for a paper discussing Chain-of-Thought reasoning."
     ```
   - Use the provided function `format_example()` to structure your input correctly and then tokenize it for the model:
     ```python
     input_text = format_example(instruction)
     inputs = tokenizer(
         input_text,
         return_tensors="pt",
         truncation=True,
         max_length=512,
         padding=True
     ).to("cuda")
     ```
   - Run the next cells to let the model generate a response:
     ```python
     with torch.no_grad():
         output_ids = model.generate(
             input_ids=inputs["input_ids"],
             attention_mask=inputs["attention_mask"],
             max_new_tokens=256
         )

     # Decode and print the response
     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
     print(generated_text)
     ```
   - The model will generate a response recommending a paper based on your query.

### 4. **Testing Different Queries**:
   - Feel free to modify the `instruction` variable with different research-related questions. For example:
     ```python
     instruction = "I need papers about machine learning optimization techniques."
     ```
   - Follow the same steps to tokenize and generate the response.

## Limitations

- Extrinsic Hallucination: During testing, we observed that the model sometimes generates hallucinated content that isn’t factual. For this type of task, we recommend using Retrieval-Augmented Generation (RAG) to improve accuracy.
- Fine-Tuning Time: The fine-tuning process can take several hours depending on your hardware capabilities, especially when dealing with larger models like LLaMA 3.1-8B.

## Future Enhancements

- RAG Integration: Future iterations of this project will likely integrate Retrieval-Augmented Generation to reduce hallucinations and improve the factual grounding of generated responses.
- Customizable API: A REST API will be implemented to make querying the model easier and to allow external applications to interact with the fine-tuned model.

## License

This project is distributed under the MIT License. Please check the LICENSE file for more information.
