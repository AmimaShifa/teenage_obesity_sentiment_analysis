#!pip install textblob openai pandas tqdm

import pandas as pd
import time
import json
import random
from textblob import TextBlob
from openai import OpenAI
from tqdm import tqdm
from json import JSONDecodeError
import os

# ---------------- CONFIG ----------------
API_KEY = "API_KEY"
input_file = "comments_file.csv"
output_file = "two_stage_sentiment_results.csv"
text_column = "comment_text"
batch_size = 25
# --------------------------------------

client = OpenAI(api_key=API_KEY)

# Load data
df = pd.read_csv(input_file, encoding="ISO-8859-1")
df = df.dropna(subset=[text_column]).reset_index(drop=True)

# Resume logic
if os.path.exists(output_file):
    result_df = pd.read_csv(output_file)
    processed_count = len(result_df)
    print(f"Resuming from row {processed_count}")
else:
    result_df = pd.DataFrame(columns=[
        "comment",
        "textblob_polarity",
        "textblob_sentiment",
        "llm_sentiment"
    ])
    processed_count = 0
    print("Starting fresh...")

# -------- STAGE 1: TEXTBLOB --------
def textblob_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0, "Neutral"

    polarity = round(TextBlob(text).sentiment.polarity, 3)

    if polarity > 0.2:
        sentiment = "Positive"
    elif polarity < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, sentiment

# -------- STAGE 2: LLM REFINEMENT --------
def backoff_retry(func, retries=3):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            wait = 10 + attempt * 5 + random.uniform(0, 5)
            print(f"Retry {attempt+1}/{retries} after error: {e}")
            time.sleep(wait)
    return None

def refine_with_llm(batch):
    prompt = (
        "You are refining sentiment analysis results.\n"
        "Each comment already has a lexicon-based sentiment from TextBlob.\n"
        "Correct it only if context, sarcasm, or nuance suggests otherwise.\n\n"
        "Return JSON format:\n"
        "[{\"sentiment\": \"Positive|Neutral|Negative\"}]\n\n"
        "Data:\n"
    )

    for i, row in enumerate(batch):
        prompt += (
            f"{i+1}. Comment: \"{row['comment']}\"\n"
            f"   TextBlob polarity: {row['polarity']}\n"
            f"   TextBlob sentiment: {row['sentiment']}\n\n"
        )

    response = backoff_retry(lambda: client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ))

    if response is None:
        raise ValueError("LLM failed after retries")

    content = response.choices[0].message.content.strip().strip("`")
    return json.loads(content)

# -------- PIPELINE --------
with tqdm(total=len(df) - processed_count, desc="Processing comments") as pbar:
    for start in range(processed_count, len(df), batch_size):
        end = min(start + batch_size, len(df))

        stage1_results = []
        for i in range(start, end):
            comment = df.loc[i, text_column]
            polarity, sentiment = textblob_sentiment(comment)

            stage1_results.append({
                "comment": comment,
                "polarity": polarity,
                "sentiment": sentiment
            })

        try:
            llm_results = refine_with_llm(stage1_results)
        except Exception:
            llm_results = [{"sentiment": "Error"}] * len(stage1_results)

        batch_rows = []
        for i in range(len(stage1_results)):
            batch_rows.append({
                "comment": stage1_results[i]["comment"],
                "textblob_polarity": stage1_results[i]["polarity"],
                "textblob_sentiment": stage1_results[i]["sentiment"],
                "llm_sentiment": llm_results[i]["sentiment"]
            })

        result_df = pd.concat([result_df, pd.DataFrame(batch_rows)], ignore_index=True)
        result_df.to_csv(output_file, index=False)

        pbar.update(len(batch_rows))
        time.sleep(1)

print("Two-stage sentiment analysis complete.")

