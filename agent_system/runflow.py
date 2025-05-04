"""
Financial Document Intelligence System - Run Flow

This module provides high-level workflows for the agents.
"""

import os
import sys
import json
import logging
import re
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from agents import Runner
import fin_assistant.config as config
from fin_assistant.agent_system.agent_definitions import (
    directory_scanner_agent,
    document_analyzer_agent,
    csv_generator_agent,
    query_agent,
    file_analyzer_agent,
    annual_report_agent,
    audit_report_agent,
    balance_sheet_agent,
    income_statement_agent,
    cash_flow_statement_agent,
    tax_document_agent,
    metadata_agent
)
from fin_assistant.agent_system.tools import (
    analyze_document_type,
    scan_directory,
    extract_annual_report_data,
    extract_audit_report_data,
    extract_balance_sheet_data,
    extract_income_statement_data,
    extract_cash_flow_statement_data,
    extract_tax_document_data,
    extract_bank_statement_data
)
from fin_assistant.utils import get_file_info, generate_id, save_json, load_json

# Set up logging
logger = logging.getLogger(__name__)

# Create a simple Context class
class Context:
    """Simple context class for agent runners"""
    
    def __init__(self):
        self._values = {}
    
    def set_value(self, key: str, value: Any):
        """Set a value in the context"""
        self._values[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the context"""
        return self._values.get(key, default)
    
    def get_values(self) -> Dict[str, Any]:
        """Get all values from the context"""
        return self._values.copy()

async def run_directory_scan(directory_path: str, recursive: bool = False) -> Dict[str, Any]:
    """
    Scan a directory for financial documents
    
    Args:
        directory_path: Path to directory
        recursive: Whether to scan recursively
        
    Returns:
        Dict with scan results
    """
    logger.info(f"Scanning directory: {directory_path}")
    
    try:
        # Call the scan_directory tool directly
        from fin_assistant.agent_system.tools import ScanDirectoryInput, scan_directory
        
        scan_input = ScanDirectoryInput(directory_path=directory_path, recursive=recursive)
        result = await scan_directory(scan_input)
        
        # For each file found, make sure we have the proper metadata structure
        if result.status == "success" and result.files_found:
            for file_info in result.files_found:
                # Generate a unique ID for each file for tracking
                file_id = f"doc_{int(time.time() * 1000)}_{hash(file_info.path) % 10000:04d}"
                
                # Generate CSV path based on file name
                file_name = os.path.basename(file_info.path)
                csv_file_name = os.path.splitext(file_name)[0] + "_extracted.csv"
                csv_file_path = os.path.join(config.CSV_DIR, csv_file_name)
                
                # Check if we need to add metadata for the file
                metadata_path = config.METADATA_PATH
                current_metadata = load_json(metadata_path, {"documents": []})
                
                # Check if the file already exists in metadata
                file_exists = False
                for doc in current_metadata.get("documents", []):
                    if doc.get("file_path") == file_info.path:
                        file_exists = True
                        # Update CSV path if needed
                        if "csv_path" not in doc or not doc["csv_path"]:
                            doc["csv_path"] = csv_file_path
                        break
                
                # If the file doesn't exist in metadata, add a basic entry
                if not file_exists:
                    basic_metadata = {
                        "id": file_id,
                        "file_name": file_name,
                        "file_path": file_info.path,
                        "file_size": file_info.size,
                        "file_extension": file_info.extension,
                        "created_at": file_info.created,
                        "modified_at": datetime.now().isoformat(),
                        "report_type": "Unknown",
                        "report_period": "Unknown",
                        "client_name": "Unknown",
                        "entity": "Unknown",
                        "account_name": "Unknown",
                        "wallet_id": f"WLT_{hash(file_info.path) % 10000:04d}",
                        "description": "File discovered during directory scan",
                        "information_present": [],
                        "csv_path": csv_file_path
                    }
                    
                    if "documents" not in current_metadata:
                        current_metadata["documents"] = []
                    
                    current_metadata["documents"].append(basic_metadata)
                    save_json(metadata_path, current_metadata)
        
        return result.model_dump()
    except Exception as e:
        logger.error(f"Error scanning directory: {str(e)}")
        return {
            "status": "error",
            "message": f"Error scanning directory: {str(e)}"
        }

async def process_document(file_path: str) -> Dict[str, Any]:
    """
    Process a document with the appropriate agent based on detected type
    
    Args:
        file_path: Path to the document
        
    Returns:
        Processed document data
    """
    # First determine document type
    try:
        analysis = await analyze_document_type(file_path)
        doc_type = analysis.document_type
        confidence = analysis.confidence
        logger.info(f"Detected report type: {doc_type} with confidence {confidence}")
        
        # Extract data based on document type
        result = {}
        try:
            if doc_type == "annual_report":
                result = await extract_annual_report_data(file_path)
            elif doc_type == "audit_report":
                result = await extract_audit_report_data(file_path)
            elif doc_type == "balance_sheet":
                result = await extract_balance_sheet_data(file_path)
            elif doc_type == "income_statement":
                result = await extract_income_statement_data(file_path)
            elif doc_type == "cash_flow_statement":
                result = await extract_cash_flow_statement_data(file_path)
            elif doc_type == "tax_document":
                result = await extract_tax_document_data(file_path)
            elif doc_type == "bank_statement":
                result = await extract_bank_statement_data(file_path)
            else:
                logger.info(f"Could not identify specific report type, using generic document analyzer")
                # Read document content
                content = ""
                file_ext = os.path.splitext(file_path)[1].lower()
                try:
                    if file_ext == ".pdf":
                        import fitz
                        doc = fitz.open(file_path)
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            content += page.get_text()
                        doc.close()
                    elif file_ext == ".docx":
                        from docx import Document
                        doc = Document(file_path)
                        for para in doc.paragraphs:
                            content += para.text + "\n"
                    elif file_ext in [".csv", ".xlsx"]:
                        import pandas as pd
                        if file_ext == ".csv":
                            df = pd.read_csv(file_path, on_bad_lines='skip')
                        else:
                            # Read all sheets and rows (first sheet only)
                            df = pd.read_excel(file_path, engine='openpyxl')
                        content = df.to_string()
                    elif file_ext == ".json":
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            content = json.dumps(data, indent=2)
                    elif file_ext == ".txt":
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                except Exception as e:
                    logger.error(f"Error reading document: {str(e)}")
                    content = ""
                
                # Create basic result
                result = {
                    "report_type": "generic_document",
                    "extracted_at": datetime.now().isoformat(),
                    "data_points": {
                        "file_name": os.path.basename(file_path),
                        "file_size": os.path.getsize(file_path),
                        "content_sample": content[:500] + "..." if len(content) > 500 else content
                    }
                }
            
            # Merge with metadata from analysis
            result["metadata"] = analysis.metadata
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing file type: {str(e)}")
            raise RuntimeError(f"Error analyzing file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise RuntimeError(f"Error processing document: {str(e)}")

async def analyze_document(file_path: str) -> Dict[str, Any]:
    """
    Analyze a document to extract its content and structure
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dict with document analysis
    """
    logger.info(f"Analyzing document: {file_path}")
    
    try:
        file_info = get_file_info(file_path)
        
        return {
            "status": "success",
            "message": f"Document {file_path} analyzed",
            "file_info": file_info,
            "document_type": "unknown",
            "extraction_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        return {
            "status": "error",
            "message": f"Error analyzing document: {str(e)}"
        }

async def generate_csv(file_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a CSV file from document analysis
    
    Args:
        file_path: Path to the source document
        data: Extracted data from the document
        
    Returns:
        Dict with CSV generation results
    """
    logger.info(f"Generating CSV for: {file_path}")
    
    try:
        # Get file name without extension
        file_name = Path(file_path).stem
        
        # Create CSV path
        csv_path = os.path.join(config.CSV_DIR, f"{file_name}_extracted.csv")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Write a simple CSV with data
        import csv
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Field", "Value"])
            
            # Write file info
            writer.writerow(["Filename", os.path.basename(file_path)])
            writer.writerow(["Path", file_path])
            writer.writerow(["Size", data.get("file_info", {}).get("size", "Unknown")])
            writer.writerow(["Created", data.get("file_info", {}).get("created", "Unknown")])
            
            # Write additional data
            writer.writerow(["Document Type", data.get("document_type", "Unknown")])
            writer.writerow(["Extraction Time", data.get("extraction_time", "Unknown")])
            
        return {
            "status": "success",
            "message": f"CSV generated at {csv_path}",
            "csv_path": csv_path
        }
    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}")
        return {
            "status": "error",
            "message": f"Error generating CSV: {str(e)}"
        }

async def process_query(query: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a query about financial data
    
    Args:
        query: User query text
        doc_id: Optional document ID to focus query on
        
    Returns:
        Dict with query processing results
    """
    from fin_assistant.agent_system.tools import answer_query
    
    logger.info(f"Processing query: {query}")
    
    try:
        # Call the answer_query tool directly
        result = await answer_query(query, doc_id)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing query: {str(e)}"
        }

async def analyze_file_type(file_path: str) -> Dict[str, Any]:
    """
    Analyze a file to determine its report type and hand off to the appropriate agent
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Dict with analysis results and next steps
    """
    logger.info(f"Analyzing file type: {file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
        
        # Ensure file_path is a string, not a Path object
        if not isinstance(file_path, str):
            file_path = str(file_path)
            
        # Call the analyze_document_type tool directly
        result = await analyze_document_type(file_path)
        
        # Extract the document type from the result
        document_type = result.document_type.lower()
        confidence = result.confidence
        
        # Map document type to report type
        report_type = None
        if "annual report" in document_type or "annual" in document_type:
            report_type = "annual_report"
        elif "audit report" in document_type or "audit" in document_type:
            report_type = "audit_report"
        elif "balance sheet" in document_type or "balance" in document_type:
            report_type = "balance_sheet"
        elif "income statement" in document_type or "income" in document_type:
            report_type = "income_statement"
        elif "cash flow" in document_type or "cash" in document_type:
            report_type = "cash_flow_statement"
        elif "tax" in document_type:
            report_type = "tax_document"
        
        # If we found a report type, process it with the appropriate agent
        if report_type and confidence >= 0.5:
            logger.info(f"Detected report type: {report_type} with confidence {confidence}")
            result_dict = await process_report(file_path, report_type)
            
            # Add analysis response to result
            result_dict["analyzer_response"] = document_type
            result_dict["confidence"] = confidence
            return result_dict
        else:
            # If we couldn't identify a specific report type, 
            # fall back to the generic document analyzer
            logger.info("Could not identify specific report type, using generic document analyzer")
            return await process_document(file_path)
        
    except Exception as e:
        logger.error(f"Error analyzing file type: {str(e)}")
        return {
            "status": "error",
            "message": f"Error analyzing file: {str(e)}"
        }

async def process_report(file_path: str, report_type: str) -> Dict[str, Any]:
    """
    Process a financial report using the appropriate report type agent
    
    Args:
        file_path: Path to the report file
        report_type: Type of report (annual_report, audit_report, etc.)
        
    Returns:
        Dict with processing results
    """
    # Import extraction tools
    from fin_assistant.agent_system.tools import (
        extract_annual_report_data,
        extract_audit_report_data,
        extract_balance_sheet_data,
        extract_income_statement_data,
        extract_cash_flow_statement_data,
        extract_tax_document_data
    )
    
    logger.info(f"Processing {report_type} from file: {file_path}")
    
    try:
        # Map report type to extraction function
        extraction_funcs = {
            "annual_report": extract_annual_report_data,
            "audit_report": extract_audit_report_data,
            "balance_sheet": extract_balance_sheet_data,
            "income_statement": extract_income_statement_data,
            "cash_flow_statement": extract_cash_flow_statement_data,
            "tax_document": extract_tax_document_data
        }
        
        # Get the appropriate extraction function
        if report_type not in extraction_funcs:
            logger.error(f"Unknown report type: {report_type}")
            return {
                "status": "error",
                "message": f"Unknown report type: {report_type}"
            }
            
        extraction_func = extraction_funcs[report_type]
        
        # Extract data using the appropriate function
        extracted_data = await extraction_func(file_path)
        
        # Generate metadata for the document
        metadata = extracted_data
        metadata["file_path"] = file_path
        metadata["report_type"] = report_type
        metadata["extraction_time"] = datetime.now().isoformat()
        
        # Save metadata to file
        metadata_dir = os.path.dirname(file_path)
        metadata_path = os.path.join(metadata_dir, "metadata.json")
        
        current_metadata = load_json(metadata_path, {})
        file_name = os.path.basename(file_path)
        
        if "documents" not in current_metadata:
            current_metadata["documents"] = {}
        
        current_metadata["documents"][file_name] = {
            "path": file_path,
            "report_type": report_type,
            "metadata": metadata
        }
        
        save_json(metadata_path, current_metadata)
        
        # Generate CSV from the extracted data
        csv_result = await generate_csv(file_path, metadata)
        
        return {
            "status": "success",
            "report_type": report_type,
            "file_path": file_path,
            "metadata": metadata,
            "csv_path": csv_result.get("csv_path"),
            "message": f"Successfully processed {report_type} from {file_path}"
        }
        
    except Exception as e:
        logger.error(f"Error processing {report_type}: {str(e)}")
        return {
            "status": "error",
            "report_type": report_type,
            "message": f"Error processing {report_type}: {str(e)}"
        }

async def extract_annual_report_data(file_path: str) -> Dict[str, Any]:
    """Extract data from an annual report"""
    logger.info(f"Processing annual report: {file_path}")
    return await analyze_document(file_path)

async def extract_audit_report_data(file_path: str) -> Dict[str, Any]:
    """Extract data from an audit report"""
    logger.info(f"Processing audit report: {file_path}")
    return await analyze_document(file_path)

async def extract_balance_sheet_data(file_path: str) -> Dict[str, Any]:
    """Extract data from a balance sheet"""
    logger.info(f"Processing balance sheet: {file_path}")
    return await analyze_document(file_path)

async def extract_income_statement_data(file_path: str) -> Dict[str, Any]:
    """Extract data from an income statement"""
    logger.info(f"Processing income statement: {file_path}")
    return await analyze_document(file_path)

async def extract_cash_flow_statement_data(file_path: str) -> Dict[str, Any]:
    """Extract data from a cash flow statement"""
    logger.info(f"Processing cash flow statement: {file_path}")
    return await analyze_document(file_path)

async def extract_tax_document_data(file_path: str) -> Dict[str, Any]:
    """Extract data from a tax document"""
    logger.info(f"Processing tax document: {file_path}")
    return await analyze_document(file_path)

async def answer_query(query: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Answer a query about financial documents
    
    Args:
        query: The query text
        doc_id: Optional document ID to focus on
        
    Returns:
        Dict with query response
    """
    logger.info(f"Answering query: {query}, doc_id={doc_id}")
    
    try:
        return {
            "status": "success",
            "query": query,
            "result": f"Answer to query: {query}",
            "sources": []
        }
    except Exception as e:
        logger.error(f"Error answering query: {str(e)}")
        return {
            "status": "error",
            "message": f"Error answering query: {str(e)}"
        } 