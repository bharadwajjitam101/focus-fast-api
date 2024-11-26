from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from groq import Groq
import os
from dotenv import load_dotenv
import json
import base64

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Base64 to JSON Converter")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class MCQ(BaseModel):
    question: str
    options: Dict[str, str]
    correct_answer: Optional[str] = None

class MCQResponse(BaseModel):
    success: bool
    questions: List[MCQ]
    total_questions: int

class TextProcessor:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    async def extract_mcqs(self, text: str) -> List[MCQ]:
        """Extract MCQs using Groq"""
        prompt = f"""Extract multiple choice questions from this text and format as JSON:
        {text}
        Format each question as:
        {{
            "question": "question text",
            "options": {{"A": "option A", "B": "option B", "C": "option C", "D": "option D"}},
            "correct_answer": "correct option letter (if provided)"
        }}"""

        response = await self.groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048
        )

        result = json.loads(response.choices[0].message.content)
        return [MCQ(**q) for q in result]

@app.post("/convert", response_model=MCQResponse)
async def convert_base64_to_json(base64_data: str = Body(...)):
    """
    Convert base64 text to JSON file
    Send a base64 encoded text string to convert it to JSON
    """
    try:
        # Decode base64
        text = base64.b64decode(base64_data).decode('utf-8')

        # Process text
        processor = TextProcessor()

        # Extract MCQs
        mcqs = await processor.extract_mcqs(text)

        return MCQResponse(
            success=True,
            questions=mcqs,
            total_questions=len(mcqs)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Check if API is running"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
