# Prompt Generation System

This project is a Prompt Generation System that allows users to input a description of their objective or task, specify scenarios and expected outputs, and generate multiple prompt options using sophisticated algorithms. The system also includes evaluation metrics to ensure the generated prompts align with the input description. 

## Features

- **User Input and Scenario Specification**: Users can input their task description and specify various scenarios.
- **Prompt Generation**: Automatically generate multiple prompt options based on the provided information.
- **Evaluation Metrics**: Check the alignment of generated prompts with the input description using cosine similarity.
- **Prompt Testing and Ranking**: Use Monte Carlo Matchmaking and the ELO rating system to compare and rank the generated prompts.
- **User Interface**: A user-friendly interface built with Streamlit.

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/MelakuAlehegn/enterprise-RAG-system-prompt-tuning.git
    cd enterprise-RAG-system-prompt-tuning
    ```

2. Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

### Jupyter Notebook

1. Navigate to the `notebooks` directory:
    ```bash
    cd notebooks
    ```

2. Run the Jupyter notebook:
    ```bash
    jupyter notebook
    ```

3. Open and run the `prompt.ipynb` notebook to generate and evaluate prompts.

### Streamlit UI

1. Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

2. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

3. Open the provided local URL in your web browser to access the UI.

## Project Structure

- `notebooks/`: Contains the Jupyter notebook for prompt generation and evaluation.
- `frontend/`: Contains the Streamlit application for the user interface.
- `requirements.txt`: Lists all the Python dependencies required for the project.
- `.env`: Environment file to store sensitive information like API keys.

## Prompt Generation Process

1. **User Input and Scenario Specification**:
    - Users provide a description of their objective or task.
    - Users specify scenarios for the prompts.

2. **Prompt Generation**:
    - The system uses OpenAI's GPT-3.5-turbo model to generate prompts based on the input and scenarios.

3. **Evaluation Metrics**:
    - Cosine similarity is used to evaluate how well the generated prompts align with the input description.

4. **Prompt Testing and Ranking**:
    - Monte Carlo Matchmaking compares the quality of the generated prompts.
    - The ELO rating system ranks the prompts based on their performance.

5. **User Interface**:
    - A Streamlit-based UI allows users to interact with the system easily.

## Technologies Used

- **Python**: Main programming language.
- **OpenAI GPT-3.5-turbo**: Model used for prompt generation.
- **FAISS**: Library for efficient similarity search.
- **Streamlit**: Framework for creating the user interface.
- **Scikit-learn**: Used for cosine similarity calculation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.