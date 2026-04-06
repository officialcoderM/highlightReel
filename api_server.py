from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

# Create the FastAPI application
app = FastAPI(
    title="Local LLM Inference API",
    description="API for serving language models locally",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
generator = None

# Request model for generate endpoint
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text to generate from")
    max_new_tokens: int = Field(100, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature (0.1 to 2.0)")

# Response model for generate endpoint
class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    timestamp: str

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Returns the status of the API server.
    Use this endpoint to verify the server is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "API server is running",
        "model_loaded": model is not None
    }

# Root endpoint
@app.get("/")
def root():
    """
    Root endpoint with basic information.
    """
    return {
        "service": "Local LLM Inference API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/", "method": "GET", "description": "Service information"},
            {"path": "/generate", "method": "POST", "description": "Generate text from prompt"}
        ]
    }

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer, generator
    print("Loading model... This may take a few minutes.")
    
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    device = torch.device("cpu")
    model.to(device)
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print("Model loaded successfully!")

# Generate endpoint
@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    """
    Generate text based on a prompt.
    """
    global generator
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Generate text
        outputs = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True
        )
        
        generated_text = outputs[0]['generated_text']
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))