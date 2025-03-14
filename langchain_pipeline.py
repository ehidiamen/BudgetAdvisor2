import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnableParallel

# Load environment variables (API keys)
load_dotenv()

# Initialize AI model (Groq Llama3)
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.1,  # More deterministic output
    max_tokens=None,  
    max_retries=2  
)

# Function to clean AI response & extract JSON
def extract_json(text):
    """Extracts valid JSON from AI response, handling both objects `{}` and lists `[]`."""
    try:
        # Directly parse if already valid JSON
        parsed_data = json.loads(text)
        return parsed_data  # Return parsed JSON directly
    except json.JSONDecodeError:
        # Handle cases where AI wraps JSON in extra text
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())  # Extract and parse valid JSON
            except json.JSONDecodeError:
                pass  # Continue to final error return

    return {"error": "Invalid JSON format from AI.", "raw_response": text}

# I Prompt for Budget Breakdown (CSV)
csv_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial expert. Convert user input into CSV format:
        Category,Item,Amount
        Income,[source],[amount]
        Expense,[category],[amount]
        Savings,[type],[amount]
        """),
    ("human", "{user_input}"),
])

# AI Prompt for Financial Advice
advice_prompt = PromptTemplate(
    template="""Based on this situation: {user_input},  
        provide clear, actionable financial advice.  
        - Budgeting  
        - Emergency Funds  
        - Debt Management  
        - Long-term savings  
        Use bullet points.""",
    input_variables=["user_input"],
)

# Chain for parallel execution of CSV & Advice
csv_advice_chain = RunnableParallel(csv=csv_prompt | llm, advice=advice_prompt | llm)

# Function to run AI-generated budget & advice
def financial_planner(user_input: str):
    if not user_input.strip():
        return {"error": "Please provide financial details."}

    result = csv_advice_chain.invoke({"user_input": user_input})
    return result["csv"].content, result["advice"].content


# =========================== JSON-Based Data Extraction =========================== #

# AI Prompts for Extracting Income, Expenses, Concerns, and Advice
income_prompt = PromptTemplate.from_template(
    "From this input: {input}, extract ONLY income sources.\n"
    "Respond ONLY in JSON: [{{\"source\": \"...\", \"amount\": ...}}]"
)

expenses_prompt = PromptTemplate.from_template(
    "From this input: {input}, extract ONLY expenses.\n"
    "Respond ONLY in JSON: [{{\"category\": \"...\", \"amount\": ...}}]"
)

concerns_prompt = PromptTemplate.from_template(
    "From this input: {input}, extract financial concerns and goals."
)

advice_prompt = PromptTemplate.from_template(
    "Based on this situation: {input}, provide actionable financial advice."
)

# Chains for processing income, expenses, concerns, and advice
income_chain = income_prompt | llm
expenses_chain = expenses_prompt | llm
concerns_chain = concerns_prompt | llm
advice_chain = advice_prompt | llm

# Optimized Parallel Execution Chain
budget_parallel_chain = RunnableParallel(
    income=income_chain,
    expenses=expenses_chain,
    concerns=concerns_chain,
    advice=advice_chain,
)

def calculate_savings(income, expenses):
    """Calculate recommended savings based on income and expenses."""
    
    # Ensure income is a valid number
    total_income = income if isinstance(income, (int, float)) else 0
    
    # Ensure expenses is a valid list and filter out None values
    total_expenses = sum(item["amount"] for item in expenses if isinstance(item.get("amount"), (int, float)))
    
    # Ensure expenses do not exceed income
    recommended_savings = max(0, total_income - total_expenses)
    
    return recommended_savings


# Function to Run Full Budget Analysis
def run_budget_pipeline(user_input):
    if not user_input.strip():
        return {"error": "Please provide financial details."}

    extracted_data = budget_parallel_chain.invoke({"input": user_input})
    
    # Validate & Parse JSON Responses
    income_data = extract_json(extracted_data["income"].content)
    expenses_data = extract_json(extracted_data["expenses"].content)
    
    if "error" in income_data or "error" in expenses_data:
        print(income_data)
        print(expenses_data)
        return {"error": "AI returned invalid financial data."}

    # Convert extracted values to structured format
    total_income = sum(item["amount"] for item in income_data)
    total_expenses = expenses_data if isinstance(expenses_data, list) else []
    recommended_savings = calculate_savings(total_income, total_expenses)

    return {
        "income": total_income,
        "expenses": total_expenses,
        "savings": recommended_savings,
        "concerns": extracted_data["concerns"],
        "advice": extracted_data["advice"]
    }
