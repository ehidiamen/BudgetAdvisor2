from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import os
import uuid  # ✅ Import UUID for unique filenames
from langchain_pipeline import run_budget_pipeline

load_dotenv()

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BudgetRequest(BaseModel):
    prompt: str  # ✅ User's budget input (e.g., "I earn $5000 and spend $2000 on rent")

@app.post("/generate_budget")
async def generate_budget(request: BudgetRequest):
    """Processes user input, generates structured budget JSON, and saves an Excel file."""
    
    # ✅ Call AI-powered pipeline to generate structured budget data
    budget_data = run_budget_pipeline(request.prompt)

    # ✅ Handle errors from AI pipeline
    if "error" in budget_data:
        return {"error": budget_data["error"]}

    # ✅ Convert structured JSON to DataFrame
    df = pd.DataFrame(budget_data["expenses"])

    # ✅ Add Income and Savings Summary
    income_row = pd.DataFrame([{"category": "Income", "amount": budget_data["income"]}])
    savings_row = pd.DataFrame([{"category": "Recommended Savings", "amount": budget_data["savings"]}])
    df = pd.concat([income_row, df, savings_row], ignore_index=True)

    # ✅ Ensure "budgets/" folder exists
    os.makedirs("budgets", exist_ok=True)

    # ✅ Generate Unique Filename (e.g., budgets/budget_abc123.xlsx)
    unique_filename = f"budget_{uuid.uuid4().hex}.xlsx"
    file_path = os.path.join("budgets", unique_filename)

    # ✅ Save the file
    df.to_excel(file_path, index=False)

    return {
        "budget": budget_data,
        "excel_url": f"/download/{unique_filename}"  # ✅ Return unique file path
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
