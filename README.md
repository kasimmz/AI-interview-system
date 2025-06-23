# Resume Matching Score Generator

A Flask web application that calculates the similarity score between a resume (PDF) and job description using natural language processing.

## Features
- Upload PDF resume
- Enter job description
- Get matching score percentage
- Responsive web interface

## How it Works
- Extracts text from PDF using PyMuPDF
- Uses SentenceTransformers for text embeddings
- Calculates cosine similarity between resume and job description
- Returns percentage match score

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python main.py`
4. Open http://localhost:5000

## Technologies Used
- Flask
- SentenceTransformers
- PyMuPDF
- scikit-learn
- HTML/CSS/JavaScript

## Live Demo
[Add your deployment URL here]