from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import warnings
import logging
import json
import psutil
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# Setup rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Local LLM Inference API",
    description="API for serving language models locally",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]
)

# Global variables for model and tokenizer
model = None
tokenizer = None
generator = None
request_count = 0
total_tokens_generated = 0

# Request model for generate endpoint
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text to generate from")
    max_new_tokens: int = Field(100, ge=1, le=500, description="Maximum number of tokens to generate (1-500)")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature (0.1 to 2.0)")

# Response model for generate endpoint
class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    timestamp: str
    generation_time_ms: float

# Metrics response model
class MetricsResponse(BaseModel):
    request_count: int
    total_tokens_generated: int
    model_loaded: bool
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    uptime_seconds: float
    start_time: str

# Health check endpoint
@app.get("/health")
@limiter.limit("10/second")
def health_check(request: Request):
    """
    Returns the status of the API server.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "API server is running",
        "model_loaded": model is not None
    }

# Root endpoint
@app.get("/")
@limiter.limit("10/second")
def root(request: Request):
    """
    Root endpoint with basic information.
    """
    return {
        "service": "Local LLM Inference API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/", "method": "GET", "description": "Service information"},
            {"path": "/generate", "method": "POST", "description": "Generate text from prompt"},
            {"path": "/metrics", "method": "GET", "description": "System and usage metrics"}
        ]
    }

# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse)
@limiter.limit("10/second")
def get_metrics(request: Request):
    """
    Returns system metrics and usage statistics.
    """
    global request_count, total_tokens_generated, start_time
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return MetricsResponse(
        request_count=request_count,
        total_tokens_generated=total_tokens_generated,
        model_loaded=model is not None,
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=psutil.virtual_memory().percent,
        memory_used_mb=memory_info.rss / 1024 / 1024,
        uptime_seconds=uptime,
        start_time=start_time.isoformat()
    )

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer, generator, start_time, request_count, total_tokens_generated
    
    start_time = datetime.now()
    request_count = 0
    total_tokens_generated = 0
    
    logger.info("Starting model loading...")
    
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
    
    logger.info("Model loaded successfully!")

# Generate endpoint
@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("5/minute")
def generate_text(request: Request, generate_request: GenerateRequest):
    """
    Generate text based on a prompt.
    Rate limited to 5 requests per minute per IP address.
    """
    global generator, request_count, total_tokens_generated
    
    client_ip = get_remote_address(request)
    logger.info(f"Generate request from {client_ip}: prompt='{generate_request.prompt[:50]}...'")
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    start_time_request = datetime.now()
    
    try:
        # Generate text
        outputs = generator(
            generate_request.prompt,
            max_new_tokens=generate_request.max_new_tokens,
            temperature=generate_request.temperature,
            do_sample=True
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Update metrics
        request_count += 1
        total_tokens_generated += generate_request.max_new_tokens
        
        generation_time = (datetime.now() - start_time_request).total_seconds() * 1000
        
        logger.info(f"Generate completed in {generation_time:.2f}ms, tokens={generate_request.max_new_tokens}")
        
        return GenerateResponse(
            prompt=generate_request.prompt,
            generated_text=generated_text,
            timestamp=datetime.now().isoformat(),
            generation_time_ms=generation_time
        )
    except Exception as e:
        logger.error(f"Generate failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))