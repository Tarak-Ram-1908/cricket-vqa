# CricketVQA

A Visual Question Answering dataset and model for T20 cricket broadcast frames.

## Overview
CricketVQA is the first VQA benchmark for cricket, covering field placement analysis in T20 matches. We fine-tune LLaVA-1.5-7B on a custom annotated dataset of 2,500 broadcast frames with 5 question types per frame.

## Question Types
- **Counting** — How many fielders are inside the 30-yard circle?
- **Spatial** — Is there a fine leg fielder in position?
- **Formation** — What field setting is this?
- **Context** — What batting phase does this field suggest?
- **Grounding** — Where is the slip cordon located?

## Project Structure
- data/ — frames and annotations
- scripts/ — frame extraction and dataset export
- src/ — model training and evaluation
- configs/ — training hyperparameters
- demo/ — Gradio inference demo
- paper/ — arXiv-style writeup

## Results
| Model | VQA Accuracy |
|---|---|
| LLaVA-1.5-7B zero-shot | ~38% |
| LLaVA-1.5-7B fine-tuned | ~68% |
| GPT-4V zero-shot (ceiling) | ~79% |

## Setup
pip install -r requirements.txt

## Author
Alahari Tarak Ram
