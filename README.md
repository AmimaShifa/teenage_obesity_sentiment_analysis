# Sentiment Analysis Phase 1 - Adolescent Obesity Perception from Social Media

This repository contains a Python script to perform sentiment analysis on social media comments using OpenAI's GPT-4o-mini model. The focus is to classify public sentiment on adolescent obesity, enabling further insights for public health research.

## 📌 Overview

This script takes a CSV file of user collected comments from social media platform, processes them in batches, and uses GPT-4 to classify each comment as **Positive**, **Negative**, or **Neutral**, along with a sentiment score between -1 and 1. The output is stored in a CSV file for downstream analysis and visualization.

## 🧠 Model Details

- **LLM Used**: `gpt-4o-mini`
- **Sentiment Rules**:
  - Score > 0.2 → **Positive**
  - Score < -0.2 → **Negative**
  - Otherwise → **Neutral**
