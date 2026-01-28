# Cross-Platform Sentiment Analysis Phase 1 - Adolescent Obesity Perception from Social Media

This repository contains a Python implementation of a **two-stage sentiment analysis pipeline** designed to analyse public perceptions of adolescent obesity from social media comments.
The methodology combines:
1) **Lexicon-based sentiment scoring (TextBlob)** for efficiency and reproducibility  
2) **Large Language Model refinement (GPT-4o-mini)** for contextual and semantic accuracy
The datasets used in this project are made publicly available to promote transparency and reproducibility in research. This repository is open source and intended for educational and research use.

A journal and conference publication based on this study is currently in preparation to formally present the methodology and findings.

---

## 📌 Overview

The pipeline processes a CSV file containing social media comments and applies **two sequential sentiment analysis stages**:

### Stage 1: Lexicon-Based Sentiment (TextBlob)
- Computes a polarity score between **-1 and +1**
- Applies fixed thresholds to classify sentiment
- Fast, deterministic, and cost-efficient
- Limited in contextual and sarcastic understanding

### Stage 2: LLM-Based Sentiment Refinement
- Uses **GPT-4o-mini** to refine or correct the initial sentiment
- Considers context, nuance, and implicit meaning
- Acts as a refinement layer, not a replacement

Final results are saved to a CSV file for downstream analysis and visualisation.

---

## 📦 Requirements

Install dependencies:

```bash
pip install --upgrade openai pandas tqdm textblob
```

---

## 🧠 Model Details
# Stage 1: TextBlob
- **Type**: Lexicon-based sentiment analysis
- **Output**:
  - Polarity score in range -1 to +1
  - Initial sentiment label
- **Threshold Rules**:
  - Polarity > 0.2 → Positive
  - Polarity < -0.2 → Negative
  - Otherwise → Neutral
    
# Stage 2: Large Language Model
- **LLM Used**: `gpt-4o-mini`
- **Role**: Contextual refinement of TextBlob output
- **Input**:
  - Original comment
  - TextBlob polarity
  - TextBlob sentiment label
- **Output**: Final refined sentiment label
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
two_stage_sentiment_results.csv
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

Output (`two_stage_sentiment_results.csv`):

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
| `output_file` | Output file name | `"two_stage_sentiment_results.csv"` |
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
- Previously processed results are read from `two_stage_sentiment_results.csv`.

---

## 📝 Example Command

```bash
python sentiment_classifier.py
```

Output:

```
Starting fresh...
Processing comments: 100%|████████████████████████████████████████| 500/500 [03:24<00:00, 2.45it/s]
All done. Output saved to two_stage_sentiment_results.csv
```

---

## 📊 Example Visualisation (Optional)

You can quickly analyse results in Python:

```python
import pandas as pd
df = pd.read_csv("two_stage_sentiment_results.csv")
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
