# Cross-Platform Sentiment Analysis Phase 1 - Adolescent Obesity Perception from Social Media

This repository contains a Python script that performs sentiment analysis on social media comments using OpenAI's GPT-4 mini model. The focus is on classifying public sentiment regarding adolescent obesity, enabling further insights for public health research.
We have made the datasets used in this project publicly available to promote transparency and collaboration in research. The datasets can be accessed through this repository, and it is open source for educational and research purposes.
Additionally, we are currently working on a journal and conference publication related to this study to further disseminate our findings and methodologies.

---

## 📌 Overview

This script takes a CSV file of user-collected comments from a social media platform, processes them in batches, and uses GPT-4 to classify each comment as **Positive**, **Negative**, or **Neutral**, along with a sentiment score between -1 and 1. The output is stored in a CSV file for downstream analysis and visualisation.

---

## 📦 Requirements

Install dependencies:

```bash
pip install --upgrade openai pandas tqdm
```

---

## 🧠 Model Details

- **LLM Used**: `gpt-4o-mini`
- **Sentiment Rules**:
  - Score > 0.2 → **Positive**
  - Score < -0.2 → **Negative**
  - Otherwise → **Neutral**


---

## ⚙️ How to Run

1. **Prepare your input file**

   Create a CSV file named `comments_file.csv` with at least one column:
   ```csv
   comment_text
   This movie was amazing!
   I didn’t like the food.
   The weather is okay today.
   ```

2. **Add your OpenAI API key**

   In the script:
   ```python
   client = OpenAI(api_key="YOUR_API_KEY_HERE")
   ```

3. **Run the script**

   ```bash
   python sentiment_classifier.py
   ```

4. **Wait for progress**

   The script processes comments in batches (default: 25) and shows progress via a tqdm bar.

---

## 🧩 Output

The script creates a file named:

```
chatgptclassified_sentiments.csv
```

Each row contains:
| comment | score | sentiment |
|----------|--------|-----------|
| "This movie was amazing!" | 0.87 | Positive |
| "I didn’t like the food." | -0.65 | Negative |
| "The weather is okay today." | 0.05 | Neutral |

- **score** ranges from **-1 to 1**  
- **sentiment** is determined by:  
  - `Positive` if score > 0.2  
  - `Negative` if score < -0.2  
  - `Neutral` otherwise  

---

## 🧰 Example Output

Input (`comments_file.csv`):

```csv
comment_text
I love this!
It’s just fine.
This is terrible.
```

Output (`chatgptclassified_sentiments.csv`):

```csv
comment,score,sentiment
I love this!,0.85,Positive
It’s just fine.,0.05,Neutral
This is terrible.,-0.78,Negative
```

---

## 🕹️ Parameters

| Variable | Description | Default |
|-----------|--------------|----------|
| `input_file` | Path to input CSV | `"comments_file.csv"` |
| `output_file` | Output file name | `"chatgptclassified_sentiments.csv"` |
| `batch_size` | Number of comments per API call | `25` |
| `retries` | API retry attempts | `3` |

---

## 🛡️ Error Handling

- If API errors or timeouts occur, the script retries with backoff.  
- If JSON decoding fails, the affected comments get labelled as:
  ```json
  {"score": null, "sentiment": "ParseError"}
  ```

---

## 💾 Resuming Progress

If the script is interrupted or crashes:
- It **automatically resumes** from where it left off.
- Previously processed results are read from `chatgptclassified_sentiments.csv`.

---

## 📝 Example Command

```bash
python sentiment_classifier.py
```

Output:

```
Starting fresh...
Processing comments: 100%|████████████████████████████████████████| 500/500 [03:24<00:00, 2.45it/s]
All done. Output saved to chatgptclassified_sentiments.csv
```

---

## 📊 Example Visualisation (Optional)

You can quickly analyse results in Python:

```python
import pandas as pd
df = pd.read_csv("chatgptclassified_sentiments.csv")
print(df['sentiment'].value_counts())
```

Output:
```
Positive    230
Neutral     185
Negative     85
```

---

## 🧩 Author

**Amima Shifa**  
Sentiment Analysis Project | 2025

---

## 🪪 License

This project is licensed under the MIT License.
