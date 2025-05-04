"""
Financial Document Intelligence System - Utilities

This module provides utility functions for the application.
"""

import os
import json
import uuid
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import aiohttp
import hashlib

# Define exports
__all__ = [
    'setup_logging',
    'load_json',
    'save_json',
    'generate_id',
    'get_file_info',
    'chunk_text',
    'call_ollama'
]

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def generate_id() -> str:
    """
    Generate a unique ID
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict with file information
    """
    path_obj = Path(file_path)
    stats = path_obj.stat()
    
    return {
        "name": path_obj.name,
        "path": str(path_obj),
        "size": stats.st_size,
        "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        "extension": path_obj.suffix.lower()
    }

def load_json(file_path: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load JSON data from a file
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if file doesn't exist
        
    Returns:
        Dict with loaded JSON data
    """
    if default is None:
        default = {}
        
    try:
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            return default
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return default

def save_json(file_path: str, data: Dict[str, Any]) -> bool:
    """
    Save JSON data to a file
    
    Args:
        file_path: Path to the JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False

def chunk_text(text: str, max_tokens: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens of overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Simple approximation: 1 token ~= 4 chars
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    
    # For very short text, just return as is
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Take a chunk of roughly max_chars
        end = min(start + max_chars, len(text))
        
        # If not at the end of text, try to break at a paragraph or sentence
        if end < len(text):
            # Try to find paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + max_chars // 2:
                end = paragraph_break + 2
            else:
                # Try to find sentence break
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + max_chars // 2:
                    end = sentence_break + 2
                else:
                    # Last resort: break at space
                    space = text.rfind(' ', start, end)
                    if space != -1 and space > start + max_chars // 2:
                        end = space + 1
        
        chunks.append(text[start:end])
        
        # Move start position for next chunk, with overlap
        start = end - overlap_chars if end - overlap_chars > start else end
        
        # Avoid getting stuck
        if start >= end:
            break
    
    return chunks

async def call_ollama(prompt: str, model: str = "llama2", timeout: int = 30, max_retries: int = 2, retry_delay: int = 2) -> str:
    """
    Call Ollama API to get a response
    
    Args:
        prompt: Prompt for the model
        model: Model to use
        timeout: Timeout for the API call
        max_retries: Maximum number of retries on timeout or loading errors
        retry_delay: Delay between retries in seconds
        
    Returns:
        Response from the model
    """
    import fin_assistant.config as config
    
    url = f"{config.OLLAMA_API}/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    logger.debug(f"Calling Ollama API with model {model} (timeout: {timeout}s, max_retries: {max_retries})")
    
    for attempt in range(max_retries + 1):
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                try:
                    # Use asyncio.wait_for to implement timeout
                    async def make_request():
                        async with session.post(url, json=data) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(f"Ollama API error: {response.status} - {error_text}")
                                
                                # Check if model is still loading - a retryable error
                                if "loading model" in error_text and attempt < max_retries:
                                    logger.warning(f"Model still loading, will retry after {retry_delay}s (attempt {attempt+1}/{max_retries+1})")
                                    await asyncio.sleep(retry_delay)
                                    return "RETRY_NEEDED"
                                    
                                return f"Error: {error_text}"
                            else:
                                result = await response.json()
                                elapsed = time.time() - start_time
                                logger.debug(f"Ollama response received in {elapsed:.2f}s")
                                return result.get("response", "")
                    
                    # Execute with timeout
                    response = await asyncio.wait_for(make_request(), timeout=timeout)
                    
                    # Check if we need to retry
                    if response == "RETRY_NEEDED":
                        continue
                        
                    elapsed = time.time() - start_time
                    logger.info(f"Ollama call completed in {elapsed:.2f}s")
                    return response
                    
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.error(f"Ollama API call timed out after {elapsed:.2f}s (limit: {timeout}s)")
                    
                    if attempt < max_retries:
                        logger.warning(f"Retrying after timeout, attempt {attempt+1}/{max_retries+1}")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    return "Error: Request timed out"
                    
        except aiohttp.ClientConnectorError:
            if attempt < max_retries:
                logger.warning(f"Connection error, retrying attempt {attempt+1}/{max_retries+1}")
                await asyncio.sleep(retry_delay)
                continue
            raise ConnectionError("Cannot connect to Ollama API. Make sure Ollama is running.")
            
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            if attempt < max_retries:
                logger.warning(f"Retrying after error, attempt {attempt+1}/{max_retries+1}")
                await asyncio.sleep(retry_delay)
                continue
            return f"Error: {str(e)}"
    
    return "Error: All retry attempts failed" 