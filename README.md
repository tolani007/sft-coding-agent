# My AI Coding Helper Training

I use this project to teach an AI how to code. I use a method called Supervised Fine-Tuning. I teach the AI using past records of how humans code.

## What is in this project

*   `scripts/`: Python code to download data, change the data format, and merge it.
*   `notebooks/`: A notebook to run the training on Google Colab.
*   `configs/`: Files that hold settings for the training.

## How I do it

1.  I download records of coding sessions.
2.  I change the format of these records so the AI can read them.
3.  I merge all the records into one big file.
4.  I push this big file to Hugging Face.
5.  I use Google Colab to train the AI with this file.
