# Generative Text Model #
This project fine-tunes a pre-trained GPT-2 model to generate coherent paragraphs on a specific topic.

# Features

-   Fine-tunes GPT-2 on a custom text dataset (`data.txt`).
-   Generates new paragraphs based on a user-provided prompt.
-   Scripts are separated for training (`train.py`) and generation (`generate.py`).
# Install dependencies
  It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
# Usage

1.  **Prepare Data:**
    Add your topic-specific text to the `data.txt` file. The more data, the better the results.

2.  **Train the Model:**
    Run the training script. This will take a while and requires a GPU.
    ```bash
    python train.py
    ```
    This will create a `fine_tuned_gpt2` directory containing your new model.

3.  **Generate Text:**
    Run the generation script with your desired starting prompt in quotes.
    ```bash
    python generate.py "The future of Mars colonization is"
    ```

# Example Output
    The mission of this program will be the first of its kind to explore the solar system, and its success is the fulfillment of NASA's mission to 
reach the stars in a matter of months and millions of years," said David Kelly, NASA Chief Scientist. "It's the culmination of over 30 years of 
research and development, leading the search for new and promising worlds."
 (NASA)

```
--- Generated Paragraph ---
