#!pip install --upgrade openai pandas tqdm
import pandas as pd
import time
import json
import random
from json import JSONDecodeError
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="API_KEY")
input_file = "comments_file.csv" 
output_file = "chatgptclassified_sentiments.csv"

df = pd.read_csv(input_file)
df = df.dropna(subset=["comment_text"]).reset_index(drop=True)

try:
    result_df = pd.read_csv(output_file)
    processed_count = result_df.shape[0]
    print(f"Resuming from row {processed_count}")
except FileNotFoundError:
    result_df = pd.DataFrame(columns=["comment", "score", "sentiment"])
    processed_count = 0
    print("Starting fresh...")
def backoff_retry(func, retries=3):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            wait = 10 + attempt * 5 + random.uniform(0, 5)
            print(f"Retry {attempt+1}/{retries} after error: {e} (waiting {wait:.1f}s)")
            time.sleep(wait)
    print("Max retries exceeded.")
    return None
def classify_sentiment_batch(batch):
    prompt = "Analyze the sentiment of the following comments. For each, return a sentiment score (-1 to 1) and a sentiment label using these rules:\n"
    prompt += "- Positive if score > 0.2\n- Negative if score < -0.2\n- Neutral otherwise\n\n"
    prompt += "Return JSON format: [{\"comment\": \"...\", \"score\": ..., \"sentiment\": \"...\"}]\n\n"
    prompt += "Comments:\n" + "\n".join([f"{i+1}. \"{c}\"" for i, c in enumerate(batch)])

    try:
        response = backoff_retry(lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        ))

        if response is None:
            raise ValueError("❌ No response after retries.")

        content = response.choices[0].message.content.strip()
        content = content.strip('` \n')
        if content.lower().startswith("json"):
            content = content[4:].strip()

        results = json.loads(content)
        return results
    except JSONDecodeError as e:
        print(f"JSON Decode Error. Raw content (first 300 chars): {content[:300]}")
        return [{"comment": c, "score": None, "sentiment": "ParseError"} for c in batch]

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return [{"comment": c, "score": None, "sentiment": "Error"} for c in batch]
batch_size = 25
with tqdm(total=len(df) - processed_count, desc="Processing comments") as pbar:
    for start in range(processed_count, len(df), batch_size):
        end = min(start + batch_size, len(df))
        comments_batch = df.loc[start:end-1, "comment_text"].tolist()

        results = classify_sentiment_batch(comments_batch)
        batch_df = pd.DataFrame(results)

        result_df = pd.concat([result_df, batch_df], ignore_index=True)
        result_df.to_csv(output_file, index=False)

        pbar.update(len(batch_df))
        time.sleep(1.0)  

print("All done. Output saved to", output_file)
