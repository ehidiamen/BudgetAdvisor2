from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from vosk import Model, KaldiRecognizer
import wave
import pandas as pd
import os
import json
import requests
import uuid  # Import UUID for unique filenames
from langchain_pipeline import run_budget_pipeline

load_dotenv()

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Deepgram API Key (Load from Environment Variable)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise Exception("⚠️ Deepgram API Key is missing! Set DEEPGRAM_API_KEY in your environment variables.")

class BudgetRequest(BaseModel):
    prompt: str  # User's budget input (e.g., "I earn $5000 and spend $2000 on rent")

# Define Request Model for Form Input
class BudgetFormRequest(BaseModel):
    income: float
    expenses: list[dict]  # Example: [{"category": "Rent", "amount": 1500}, ...]
    concerns: str = ""  # Optional financial concerns


@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receives an audio file and sends it to Deepgram for transcription.
    Returns the transcribed text.
    """

    # Read the uploaded file into memory
    audio_bytes = await file.read()

    # Deepgram API URL
    url = "https://api.deepgram.com/v1/listen"

    # Headers for Deepgram API
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": file.content_type,  # Automatically detects file type
    }

    # Send Audio to Deepgram
    response = requests.post(url, headers=headers, data=audio_bytes)

    # Check for errors
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Deepgram Error: {response.json()}")

    # Extract Transcription
    transcript_data = response.json()
    transcript_text = transcript_data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")

    return {"transcription": transcript_text}

@app.post("/generate_budget_from_form")
async def generate_budget_from_form(request: BudgetFormRequest):
    """Processes structured budget data from the form, generates an AI-enhanced budget, and saves an Excel file."""
    
    # Convert form input into a prompt format for AI pipeline
    user_input = f"My monthly income is ${request.income}. "
    
    if request.expenses:
        user_input += "I have the following expenses: "
        user_input += ", ".join([f"{exp['category']} (${exp['amount']})" for exp in request.expenses]) + ". "

    if request.concerns:
        user_input += f"My financial concerns are: {request.concerns}. "

    # Send structured input to AI pipeline
    budget_data = run_budget_pipeline(user_input)

    # Handle errors from AI pipeline
    if "error" in budget_data:
        return {"error": budget_data["error"]}

    # Convert structured JSON to DataFrame
    df = pd.DataFrame(budget_data["expenses"])

    # Add Income and Savings Summary
    income_row = pd.DataFrame([{"category": "Income", "amount": budget_data["income"]}])
    savings_row = pd.DataFrame([{"category": "Recommended Savings", "amount": budget_data["savings"]}])
    df = pd.concat([income_row, df, savings_row], ignore_index=True)

    # Ensure "budgets/" folder exists
    os.makedirs("budgets", exist_ok=True)

    # Generate Unique Filename (e.g., budgets/budget_abc123.xlsx)
    unique_filename = f"budget_{uuid.uuid4().hex}.xlsx"
    file_path = os.path.join("budgets", unique_filename)

    # Save the file
    df.to_excel(file_path, index=False)

    return {
        "budget": budget_data,
        "excel_url": f"/download/{unique_filename}"  # Return unique file path
    }

@app.post("/generate_budget")
async def generate_budget(request: BudgetRequest):
    """Processes user input, generates structured budget JSON, and saves an Excel file."""
    
    # Call AI-powered pipeline to generate structured budget data
    budget_data = run_budget_pipeline(request.prompt)

    # Handle errors from AI pipeline
    if "error" in budget_data:
        return {"error": budget_data["error"]}

    # Convert structured JSON to DataFrame
    df = pd.DataFrame(budget_data["expenses"])

    # Add Income and Savings Summary
    income_row = pd.DataFrame([{"category": "Income", "amount": budget_data["income"]}])
    savings_row = pd.DataFrame([{"category": "Recommended Savings", "amount": budget_data["savings"]}])
    df = pd.concat([income_row, df, savings_row], ignore_index=True)

    # Ensure "budgets/" folder exists
    os.makedirs("budgets", exist_ok=True)

    # Generate Unique Filename (e.g., budgets/budget_abc123.xlsx)
    unique_filename = f"budget_{uuid.uuid4().hex}.xlsx"
    file_path = os.path.join("budgets", unique_filename)

    # Save the file
    df.to_excel(file_path, index=False)

    return {
        "budget": budget_data,
        "excel_url": f"/download/{unique_filename}"  # Return unique file path
    }

@app.get("/download/{filename}")
async def download_budget(filename: str):
    """Serves the requested budget Excel file for download."""
    file_path = os.path.join("budgets", filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    return {"error": "File not found"}
