import os
import torch
import uvicorn
from typing import Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from accelerate import Accelerator
import logging
import mlflow
import tempfile
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_default_base_model(flow_type: str, training_type: str) -> str:
    """Get default base model based on flow and training type."""
    if flow_type == "codegen":
        return "Qwen/Qwen2.5-Coder-3B"
    elif flow_type == "idp":
        return "Qwen/Qwen2.5-3B-Instruct"
    return "Qwen/Qwen2.5-3B-Instruct"

def format_prompt(input_text: str, flow_type: str, training_type: str) -> str:
    """Format prompt based on flow and training type."""
    if flow_type == "idp":
        return f"Extract structured fields as strict JSON from the document below.\n\n<document>\n{input_text}\n</document>\nReturn only JSON."
    elif flow_type == "codegen":
        return f"You are a helpful coding assistant. Solve the task.\n\nProblem:\n{input_text}\n\nProduce correct, runnable code."
    return input_text

def download_model_from_mlflow(model_uri: str, local_path: str) -> str:
    """Download model from MLflow."""
    logger.info(f"Downloading model from MLflow URI: {model_uri}")
    try:
        os.makedirs(local_path, exist_ok=True)
        mlflow.pytorch.load_model(model_uri, dst_path=local_path)
        logger.info(f"Successfully downloaded model to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download model from MLflow: {e}")
        raise

def load_model_and_tokenizer(model_path: str, base_model_name: str) -> Tuple[PeftModel, AutoTokenizer]:
    """Load trained model and tokenizer."""
    logger.info(f"Loading tokenizer from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    except Exception:
        logger.warning(f"Tokenizer not found at {model_path}, loading from base model {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading PEFT model from {model_path} with base {base_model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    logger.info(f"Successfully loaded model and tokenizer from {model_path}")
    return model, tokenizer

# --- Pydantic Models ---

class LoadModelRequest(BaseModel):
    flow_type: str  # "idp" or "codegen"
    training_type: str  # "sft", "kd", or "dpo"
    run_id: str  # MLflow run ID

class IDPRequest(BaseModel):
    document: str
    schema: Optional[str] = None
    run_id: Optional[str] = None  # Optional run ID for dynamic loading

class CodeGenRequest(BaseModel):
    prompt: str
    run_id: Optional[str] = None  # Optional run ID for dynamic loading

class PredictionResponse(BaseModel):
    model_type: str
    run_id: Optional[str] = None
    generated_text: str
    load_time: Optional[float] = None

class ModelStatusResponse(BaseModel):
    loaded_models: Dict[str, Dict]
    message: str

# --- FastAPI Application ---

app = FastAPI(
    title="Dynamic Model Server",
    description="API for serving IDP and CodeGen models with dynamic loading",
    version="1.0.0"
)

# Global dictionary to store loaded models
loaded_models: Dict[str, Dict] = {}
model_lock = threading.Lock()  # For thread-safe model loading

def get_model_key(flow_type: str, training_type: str, run_id: str) -> str:
    """Generate a unique key for a model."""
    return f"{flow_type}_{training_type}_{run_id[:8]}"  # Use first 8 chars of run ID

@app.on_event("startup")
def startup_event():
    """Initialize MLflow connection and load any pre-configured models."""
    logger.info("Starting up model server...")
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Connected to MLflow at {mlflow_uri}")
    else:
        logger.warning("MLFLOW_TRACKING_URI not set. Dynamic loading will not work.")

@app.post("/load_model", response_model=ModelStatusResponse, tags=["Model Management"])
def load_model(request: LoadModelRequest):
    """Load a model dynamically from MLflow using run ID."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise HTTPException(status_code=500, detail="MLFLOW_TRACKING_URI not configured")
    
    key = get_model_key(request.flow_type, request.training_type, request.run_id)
    
    with model_lock:
        # Check if model is already loaded
        if key in loaded_models:
            return ModelStatusResponse(
                loaded_models=loaded_models,
                message=f"Model {key} is already loaded"
            )
        
        try:
            start_time = time.time()
            
            # Construct model URI
            model_uri = f"runs:/{request.run_id}/model"
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix=f"model_{request.run_id[:8]}_")
            
            # Download model
            download_model_from_mlflow(model_uri, temp_dir)
            
            # Get base model
            base_model = get_default_base_model(request.flow_type, request.training_type)
            
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(temp_dir, base_model)
            
            # Store in memory
            loaded_models[key] = {
                "model": model,
                "tokenizer": tokenizer,
                "flow_type": request.flow_type,
                "training_type": request.training_type,
                "run_id": request.run_id,
                "temp_dir": temp_dir,
                "load_time": time.time() - start_time,
                "loaded_at": time.time()
            }
            
            load_time = time.time() - start_time
            
            return ModelStatusResponse(
                loaded_models={k: {
                    "flow_type": v["flow_type"],
                    "training_type": v["training_type"],
                    "run_id": v["run_id"],
                    "loaded_at": v["loaded_at"]
                } for k, v in loaded_models.items()},
                message=f"Successfully loaded model {key} in {load_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model {key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.delete("/unload_model/{model_key}", response_model=ModelStatusResponse, tags=["Model Management"])
def unload_model(model_key: str):
    """Unload a model to free up memory."""
    with model_lock:
        if model_key not in loaded_models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {model_key} not found"
            )
        
        try:
            # Clean up temporary directory
            temp_dir = loaded_models[model_key].get("temp_dir")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
            # Remove from memory
            del loaded_models[model_key]
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return ModelStatusResponse(
                loaded_models={k: {
                    "flow_type": v["flow_type"],
                    "training_type": v["training_type"],
                    "run_id": v["run_id"],
                    "loaded_at": v["loaded_at"]
                } for k, v in loaded_models.items()},
                message=f"Successfully unloaded model {model_key}"
            )
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.get("/", tags=["General"])
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "loaded_models": {
            k: {
                "flow_type": v["flow_type"],
                "training_type": v["training_type"],
                "run_id": v["run_id"],
                "loaded_at": v["loaded_at"],
                "load_time": v.get("load_time", 0)
            } for k, v in loaded_models.items()
        }
    }

@app.post("/idp/predict", response_model=PredictionResponse, tags=["IDP"])
def predict_idp(request: IDPRequest):
    """Prediction endpoint for IDP models."""
    # Determine which model to use
    if request.run_id:
        # Use specific run ID
        key = get_model_key("idp", "sft", request.run_id)  # Default to sft for run_id based
        if key not in loaded_models:
            # Try to load it dynamically
            try:
                load_request = LoadModelRequest(
                    flow_type="idp",
                    training_type="sft",  # Default
                    run_id=request.run_id
                )
                load_model(load_request)
            except Exception as e:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Could not load model for run_id {request.run_id}: {str(e)}"
                )
    else:
        # Find any loaded IDP model
        idp_models = [k for k in loaded_models.keys() if k.startswith("idp_")]
        if not idp_models:
            raise HTTPException(
                status_code=404, 
                detail="No IDP models loaded. Use /load_model to load one first."
            )
        key = idp_models[0]  # Use first available
    
    model_data = loaded_models[key]
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    try:
        start_time = time.time()
        prompt = format_prompt(request.document, "idp", model_data["training_type"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt):].strip()
        
        return PredictionResponse(
            model_type=key,
            run_id=model_data["run_id"],
            generated_text=response_text,
            load_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error during IDP prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.post("/codegen/predict", response_model=PredictionResponse, tags=["CodeGen"])
def predict_codegen(request: CodeGenRequest):
    """Prediction endpoint for CodeGen models."""
    # Determine which model to use
    if request.run_id:
        # Use specific run ID
        key = get_model_key("codegen", "sft", request.run_id)  # Default to sft for run_id based
        if key not in loaded_models:
            # Try to load it dynamically
            try:
                load_request = LoadModelRequest(
                    flow_type="codegen",
                    training_type="sft",  # Default
                    run_id=request.run_id
                )
                load_model(load_request)
            except Exception as e:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Could not load model for run_id {request.run_id}: {str(e)}"
                )
    else:
        # Find any loaded CodeGen model
        codegen_models = [k for k in loaded_models.keys() if k.startswith("codegen_")]
        if not codegen_models:
            raise HTTPException(
                status_code=404, 
                detail="No CodeGen models loaded. Use /load_model to load one first."
            )
        key = codegen_models[0]  # Use first available
    
    model_data = loaded_models[key]
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    try:
        start_time = time.time()
        prompt = format_prompt(request.prompt, "codegen", model_data["training_type"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt):].strip()
        
        return PredictionResponse(
            model_type=key,
            run_id=model_data["run_id"],
            generated_text=response_text,
            load_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error during CodeGen prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)