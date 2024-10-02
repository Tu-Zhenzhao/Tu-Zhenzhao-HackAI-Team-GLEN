# Project Name
Research paper query engine for use with Wolfram Language.

## Description
We used a fine-tuning model to create a AI chatbot for querying arxiv research papers. We have automatically downloaded 80 academic papers. Within a month we learned how to use AI Workbench.

## Instructions for Starting the Program and Asking Questions

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
   - After cloning the project repository, navigate to the project directory:
     ```bash
     cd /path/to/project
     ```
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
