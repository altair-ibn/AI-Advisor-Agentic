"""
Financial Document Intelligence System - Tornado WebSocket Implementation

This module provides a complete Tornado-based web server with native WebSocket support.
It relies on the OpenAI Agents SDK for document processing and querying.
"""

import os
import json
import time
import uuid
import csv
import logging
import asyncio
from datetime import datetime
import sys
from pathlib import Path

import tornado.web
import tornado.ioloop
import tornado.websocket
import tornado.httpserver
from tornado.options import define, options

# Import fin_assistant package
import fin_assistant
import fin_assistant.config

# Import agent system
try:
    from fin_assistant.agent_system import (
        process_document,
        process_query, 
        initialize_agents, 
        generate_csv,
        run_directory_scan,
        analyze_file_type
    )
    from fin_assistant.agent_system.tools import (
        FileInfo,
        Analysis,
        ExtractTextInput,
        ExtractTextOutput,
        AnalyzeDocumentInput,
        AnalyzeDocumentOutput
    )
    logging.info("Agent system loaded successfully")
except ImportError as e:
    logging.critical(f"Failed to import agent system: {e}")
    logging.critical("This application requires the OpenAI Agents SDK. Exiting.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define command line options
define("port", default=8081, help="run on the given port", type=int)
define("debug", default=True, help="run in debug mode", type=bool)

# Application Constants
APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
UPLOADS_DIR = DATA_DIR / "uploads"
CSVS_DIR = DATA_DIR / "csvs"
METADATA_PATH = DATA_DIR / "metadata.json"
ACCEPTED_EXTENSIONS = ['.csv', '.pdf', '.txt', '.docx', '.xlsx', '.json']

# Ensure directories exist
for dir_path in [DATA_DIR, SAMPLES_DIR, UPLOADS_DIR, CSVS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    logger.critical("OPENAI_API_KEY environment variable not set")
    logger.critical("Please set your OpenAI API key as an environment variable and restart")
    sys.exit(1)

class BaseHandler(tornado.web.RequestHandler):
    """Base handler with common functionality"""
    
    def set_default_headers(self):
        """Set CORS headers"""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type,Authorization")
        self.set_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    
    def options(self, *args, **kwargs):
        """Handle preflight requests"""
        self.set_status(204)
        self.finish()

class MainHandler(BaseHandler):
    """Handler for the root URL"""
    
    def get(self):
        self.render("websocket.html")

class StatusHandler(BaseHandler):
    """API endpoint for status check"""
    
    def get(self):
        """Return API status"""
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({
            "status": "success",
            "message": "Financial Document Intelligence System API is running",
            "version": "1.0"
        }))

class DocumentsHandler(BaseHandler):
    """API endpoint for retrieving document metadata"""
    
    def get(self):
        """Return document metadata"""
        self.set_header("Content-Type", "application/json")
        
        try:
            # Load metadata from file
            if METADATA_PATH.exists():
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"documents": []}
            
            # Add CSV path to documents
            for doc in metadata.get("documents", []):
                basename = os.path.splitext(doc.get("name", ""))[0]
                doc["csv_path"] = f"{basename}_extracted.csv"
            
            self.write(json.dumps({
                "status": "success",
                "documents": metadata.get("documents", []),
                "count": len(metadata.get("documents", []))
            }))
        except Exception as e:
            logger.exception(f"Error loading metadata: {str(e)}")
            self.write(json.dumps({
                "status": "error",
                "message": f"Error loading metadata: {str(e)}"
            }))

class DocumentHandler(BaseHandler):
    """API endpoint for retrieving a specific document's metadata"""
    
    def get(self, doc_id):
        """Return document metadata for a specific document"""
        self.set_header("Content-Type", "application/json")
        
        try:
            # Load metadata from file
            if METADATA_PATH.exists():
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"documents": []}
            
            # Find document by ID
            for doc in metadata.get("documents", []):
                if doc.get("id") == doc_id:
                    # Add CSV path
                    basename = os.path.splitext(doc.get("name", ""))[0]
                    doc["csv_path"] = f"{basename}_extracted.csv"
                    
                    self.write(json.dumps({
                        "status": "success",
                        "document": doc
                    }))
                    return
            
            # Document not found
            self.set_status(404)
            self.write(json.dumps({
                "status": "error",
                "message": f"Document not found with ID: {doc_id}"
            }))
        except Exception as e:
            logger.exception(f"Error retrieving document: {str(e)}")
            self.write(json.dumps({
                "status": "error",
                "message": f"Error retrieving document: {str(e)}"
            }))

class CSVHandler(tornado.web.StaticFileHandler):
    """Handler for serving CSV files"""
    
    def set_default_headers(self):
        """Set CORS headers"""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type,Authorization")
        self.set_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    
    def validate_absolute_path(self, root, absolute_path):
        """Validate and normalize the absolute path"""
        # Make sure the path exists and is a file
        if not os.path.exists(absolute_path):
            logger.error(f"CSV file not found: {absolute_path}")
            raise tornado.web.HTTPError(404, f"CSV file not found: {os.path.basename(absolute_path)}")
        
        if not os.path.isfile(absolute_path):
            logger.error(f"Not a file: {absolute_path}")
            raise tornado.web.HTTPError(403, f"Not a file: {os.path.basename(absolute_path)}")
            
        # Make sure it's actually a CSV file
        filename = os.path.basename(absolute_path)
        _, ext = os.path.splitext(filename)
        if ext.lower() != '.csv':
            logger.error(f"Not a CSV file: {filename}")
            raise tornado.web.HTTPError(403, f"Not a CSV file: {filename}")
        
        # Make sure it's within our CSVs directory for security
        csv_dir = os.path.abspath(str(CSVS_DIR))
        file_dir = os.path.dirname(os.path.abspath(absolute_path))
        if not file_dir.startswith(csv_dir):
            logger.error(f"Security violation: {absolute_path} is outside CSV directory")
            raise tornado.web.HTTPError(403)
            
        logger.info(f"Serving CSV file: {absolute_path}")
        return super(CSVHandler, self).validate_absolute_path(root, absolute_path)
        
    def get(self, path, include_body=True):
        """Serve a CSV file"""
        try:
            # Add specific headers for CSV files
            self.set_header("Content-Type", "text/csv")
            self.set_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
            return super(CSVHandler, self).get(path, include_body)
        except Exception as e:
            logger.error(f"Error serving CSV file: {str(e)}")
            if not self._headers_written:
                self.set_status(500)
                self.write(json.dumps({
                    "status": "error",
                    "message": f"Error serving CSV file: {str(e)}"
                }))
                return

class DocumentDownloadHandler(BaseHandler):
    """Handler for document downloads"""
    
    def get(self, doc_id):
        """Download a document file"""
        try:
            # Load metadata to get file info
            if METADATA_PATH.exists():
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
            else:
                self.set_status(404)
                self.write(json.dumps({
                    "status": "error",
                    "message": "No documents available"
                }))
                return
            
            # Find document by ID
            found = False
            for doc in metadata.get("documents", []):
                if doc.get("id") == doc_id:
                    found = True
                    file_path = doc.get("path", "")
                    
                    # Create Path object and check if it exists
                    path_obj = Path(file_path)
                    logger.info(f"Attempting to download file: {file_path}")
                    
                    if path_obj.exists():
                        filename = os.path.basename(file_path)
                        
                        # Set content type based on file extension
                        content_type = self.get_content_type(path_obj)
                        if content_type:
                            self.set_header("Content-Type", content_type)
                        
                        # Set headers for download
                        self.set_header("Content-Disposition", f'attachment; filename="{filename}"')
                        
                        # Send file
                        with open(file_path, 'rb') as f:
                            self.write(f.read())
                        self.finish()
                        logger.info(f"File successfully downloaded: {file_path}")
                    else:
                        # File not found, check if it's a relative path issue
                        # Try alternative paths
                        alternate_paths = [
                            # Current path
                            path_obj,
                            # Relative to data/samples
                            SAMPLES_DIR / os.path.basename(file_path),
                            # Absolute path
                            Path(file_path)
                        ]
                        
                        for alt_path in alternate_paths:
                            logger.info(f"Trying alternate path: {alt_path}")
                            if alt_path.exists():
                                filename = os.path.basename(str(alt_path))
                                
                                # Set content type based on file extension
                                content_type = self.get_content_type(alt_path)
                                if content_type:
                                    self.set_header("Content-Type", content_type)
                                
                                # Set headers for download
                                self.set_header("Content-Disposition", f'attachment; filename="{filename}"')
                                
                                # Send file
                                with open(alt_path, 'rb') as f:
                                    self.write(f.read())
                                self.finish()
                                logger.info(f"File successfully downloaded using alternate path: {alt_path}")
                                return
                        
                        # If we get here, none of the paths worked
                        logger.error(f"File not found at path: {file_path}")
                        logger.error(f"Tried alternate paths: {alternate_paths}")
                        self.set_status(404)
                        self.write(json.dumps({
                            "status": "error",
                            "message": f"File not found: {file_path}. Please check if the file exists in the samples directory."
                        }))
                    break
            
            if not found:
                self.set_status(404)
                self.write(json.dumps({
                    "status": "error",
                    "message": f"Document not found with ID: {doc_id}"
                }))
        except Exception as e:
            logger.exception(f"Error downloading document: {str(e)}")
            self.set_status(500)
            self.write(json.dumps({
                "status": "error",
                "message": f"Error downloading document: {str(e)}"
            }))
    
    def get_content_type(self, file_path):
        """Determine content type based on file extension"""
        extension = os.path.splitext(file_path)[1].lower()
        
        content_types = {
            '.pdf': 'application/pdf',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.json': 'application/json'
        }
        
        return content_types.get(extension)

# Add the missing AgentSystem class
class AgentSystem:
    """
    Interface to the agent system functions.
    This class serves as a wrapper to interact with the agent tools.
    """
    
    def __init__(self):
        """Initialize the agent system interface"""
        logger.info("Initializing AgentSystem interface")
        self.initialized = True
    
    async def process_document(self, file_path):
        """Process a document using the agent system"""
        return await fin_assistant.agent_system.process_document(file_path)
    
    async def process_query(self, query, doc_id=None):
        """Process a query using the agent system"""
        return await fin_assistant.agent_system.process_query(query, doc_id)
    
    async def analyze_file_type(self, file_path):
        """Analyze a file type using the agent system"""
        return await fin_assistant.agent_system.analyze_file_type(file_path)
    
    async def generate_csv(self, file_path, data):
        """Generate a CSV file using the agent system"""
        return await fin_assistant.agent_system.generate_csv(file_path, data)
    
    async def scan_directory(self, directory_path, recursive=False):
        """Scan a directory using the agent system"""
        return await fin_assistant.agent_system.run_directory_scan(directory_path, recursive)

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    """Handler for WebSocket connections"""
    
    clients = set()
    
    def check_origin(self, origin):
        """Allow connections from any origin"""
        return True
    
    def initialize(self):
        self.agent_system = AgentSystem()
        self.session_data = {
            "processed_files": [],
            "status": "idle",
            "current_file": None,
            "errors": [],
            "any_successful": False,
            "progress": {
                "total_files": 0,
                "processed_files": 0,
                "current_file_progress": 0,
                "status_message": "Ready"
            }
        }
    
    def open(self):
        """Called when a WebSocket connection is established"""
        logger.info("WebSocket connection opened")
        WebSocketHandler.clients.add(self)
        
        # Send welcome message
        self.write_message(json.dumps({
            "type": "message",
            "data": "Welcome! WebSocket connection established to Financial Document Intelligence System."
        }))
    
    def on_close(self):
        """Called when a WebSocket connection is closed"""
        logger.info("WebSocket connection closed")
        if self in WebSocketHandler.clients:
            WebSocketHandler.clients.remove(self)
    
    async def on_message(self, message):
        """Called when a message is received from the client"""
        logger.info(f"Received message: {message}")
        
        try:
            # Parse the message as JSON
            data = json.loads(message)
            message_type = data.get("type", "")
            
            # Handle different message types
            if message_type == "scan_request":
                await self.handle_scan_request(data)
            elif message_type == "query":
                await self.handle_query(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                self.write_message(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
            self.write_message(json.dumps({
                "type": "error",
                "message": "Invalid JSON message"
            }))
        except Exception as e:
            logger.exception(f"Error processing message: {str(e)}")
            self.write_message(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
    
    async def handle_scan_request(self, data):
        """Handle a scan request from the client"""
        logger.info(f"Processing scan request: {data}")
        
        # Get actual files from samples directory
        sample_files = self.get_actual_files_from_directory(SAMPLES_DIR)
        
        if not sample_files:
            logger.warning(f"No files found in {SAMPLES_DIR}. Please add files to the samples directory.")
            # Return empty response
            self.write_message(json.dumps({
                "type": "scan_results",
                "data": {
                    "status": "warning",
                    "message": "No files found in samples directory",
                    "files": [],
                }
            }))
            return
        
        # Send start message
        self.write_message(json.dumps({
            "type": "scan_progress",
            "data": {
                "status": "started",
                "message": "Starting scan...",
                "progress": 0,
                "total": len(sample_files),
                "timestamp": time.time()
            }
        }))
        
        # Process each sample file
        documents = []
        any_successful = False  # Track if at least one document was processed successfully
        
        for i, file_info in enumerate(sample_files, 1):
            # Send progress update
            self.write_message(json.dumps({
                "type": "scan_progress",
                "data": {
                    "status": "in_progress",
                    "message": f"Processing file {i} of {len(sample_files)}...",
                    "progress": i,
                    "total": len(sample_files),
                    "current_file": file_info["name"],
                    "timestamp": time.time()
                }
            }))
            
            # Process the file
            doc_id = str(uuid.uuid4())
            file_path = Path(file_info["path"])
            
            try:
                # Process the document using the file analyzer agent first
                logger.info(f"Analyzing document type with agent system: {file_path}")
                result = await analyze_file_type(file_path)
                
                if result["status"] == "success":
                    # Document processed successfully
                    report_type = result.get("report_type", "unknown")
                    metadata = result.get("metadata", {})
                    csv_path = result.get("csv_path")
                    
                    # Create document metadata for response
                    doc_metadata = {
                        "id": doc_id,
                        "filename": os.path.basename(str(file_path)),
                        "path": str(file_path),
                        "size": os.path.getsize(str(file_path)),
                        "upload_time": datetime.now().isoformat(),
                        "report_type": report_type,
                        "report_period": metadata.get("report_period", "Unknown"),
                        "client_name": metadata.get("client_name", "Unknown"),
                        "entity": metadata.get("entity", "Unknown"),
                        "account_name": metadata.get("account_name", "Unknown"),
                        "wallet_id": metadata.get("wallet_id", f"WLT_{hash(str(file_path)) % 10000:04d}"),
                        "description": metadata.get("description", "No description available"),
                        "information_present": metadata.get("information_present", []),
                        "csv_path": csv_path
                    }
                    
                    # Ensure required fields are present and non-empty
                    if not doc_metadata["report_period"] or doc_metadata["report_period"] == "Unknown":
                        doc_metadata["report_period"] = "Q4 2024"  # Default meaningful value
                        
                    if not doc_metadata["description"] or doc_metadata["description"] == "No description available":
                        if doc_metadata["report_type"] == "balance_sheet":
                            doc_metadata["description"] = "Balance sheet showing assets, liabilities and equity"
                        elif doc_metadata["report_type"] == "income_statement":
                            doc_metadata["description"] = "Income statement showing revenue, expenses and profit"
                        elif doc_metadata["report_type"] == "cash_flow_statement":
                            doc_metadata["description"] = "Cash flow statement showing operating, investing and financing activities"
                        elif doc_metadata["report_type"] == "annual_report":
                            doc_metadata["description"] = "Annual report with financial performance and company information"
                        else:
                            doc_metadata["description"] = f"Financial document of type {doc_metadata['report_type']}"
                    
                    if not doc_metadata["information_present"] or len(doc_metadata["information_present"]) == 0:
                        # Generate default values based on document type
                        if doc_metadata["report_type"] == "balance_sheet":
                            doc_metadata["information_present"] = ["assets", "liabilities", "equity", "total assets", "total liabilities"]
                        elif doc_metadata["report_type"] == "income_statement":
                            doc_metadata["information_present"] = ["revenue", "expenses", "net income", "profit", "loss"]
                        elif doc_metadata["report_type"] == "cash_flow_statement":
                            doc_metadata["information_present"] = ["operating activities", "investing activities", "financing activities", "cash balance"]
                        elif doc_metadata["report_type"] == "annual_report":
                            doc_metadata["information_present"] = ["financial summary", "performance", "outlook", "company information"]
                        else:
                            doc_metadata["information_present"] = ["financial data", "analysis"]
                    
                    self.write_message(json.dumps({
                        "type": "document_processed",
                        "status": "success",
                        "document": doc_metadata,
                        "csv_path": csv_path,
                        "message": f"Document processed as {report_type}"
                    }))
                    
                    # Add document to session data
                    if "documents" not in self.session_data:
                        self.session_data["documents"] = {}
                    
                    self.session_data["documents"][doc_id] = doc_metadata
                    self.save_session()
                    
                    # Mark that at least one document was processed successfully
                    any_successful = True
                    
                    # Add to documents list for metadata saving
                    documents.append(doc_metadata)
                else:
                    # Document processing failed
                    error_message = result.get("message", "Unknown error")
                    logger.error(f"Error processing document {file_path}: {error_message}")
                    self.write_message(json.dumps({
                        "type": "document_processed",
                        "status": "error",
                        "error": error_message
                    }))
                
            except Exception as e:
                error_message = str(e)
                if "Ollama service is disabled" in error_message:
                    error_message = "Ollama service is disabled. Please enable Ollama or use OpenAI backend."
                elif "Cannot connect to Ollama API" in error_message:
                    error_message = "Cannot connect to Ollama API. Please make sure Ollama is running."
                    
                logger.exception(f"Error processing document {file_path}: {error_message}")
                self.write_message(json.dumps({
                    "type": "document_processed",
                    "status": "error",
                    "error": error_message
                }))
        
        # Save metadata if we have documents
        if documents:
            await self.save_metadata(documents)
        
        # Send completion message
        status = "completed" if any_successful else "error"
        message = "Scan completed successfully" if any_successful else "Scan completed with errors. No documents were processed."
        
        self.write_message(json.dumps({
            "type": "scan_progress",
            "data": {
                "status": status,
                "message": message,
                "progress": len(sample_files),
                "total": len(sample_files),
                "timestamp": time.time()
            }
        }))
        
        # Send scan results
        self.write_message(json.dumps({
            "type": "scan_results",
            "data": {
                "status": "success" if any_successful else "error",
                "message": "Scan completed" if any_successful else "Scan completed with errors. No documents were processed.",
                "files": sample_files
            }
        }))
    
    async def handle_query(self, data):
        """Handle a query from the client"""
        query_text = data.get("query", "")
        doc_id = data.get("doc_id")  # Accept an optional doc_id parameter
        
        if not query_text:
            self.write_message(json.dumps({
                "type": "query_result",
                "data": {
                    "status": "error",
                    "message": "No query provided"
                }
            }))
            return
        
        # Load metadata to access document information
        documents = []
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                    documents = metadata.get("documents", [])
            except Exception as e:
                logger.exception(f"Error loading metadata for query: {str(e)}")
                self.write_message(json.dumps({
                    "type": "query_result",
                    "data": {
                        "status": "error",
                        "message": f"Error loading document metadata: {str(e)}"
                    }
                }))
                return
        
        if not documents:
            self.write_message(json.dumps({
                "type": "query_result",
                "data": {
                    "status": "error",
                    "message": "No documents available to query. Please scan documents first."
                }
            }))
            return
        
        try:
            # Process query with agent system
            logger.info(f"Querying with agent system: {query_text}")
            result = await process_query(query_text, doc_id)
            
            if result["status"] == "success":
                # Send successful response
                self.write_message(json.dumps({
                    "type": "query_result",
                    "data": {
                        "status": "success",
                        "query": query_text,
                        "result": result["result"],
                        "source_documents": result.get("sources", [])
                    }
                }))
            else:
                # Handle error in agent processing
                logger.error(f"Error in agent processing: {result.get('message')}")
                self.write_message(json.dumps({
                    "type": "query_result",
                    "data": {
                        "status": "error",
                        "message": f"Error processing query: {result.get('message', 'Unknown error')}"
                    }
                }))
                
        except Exception as e:
            error_message = str(e)
            if "Ollama service is disabled" in error_message:
                error_message = "Ollama service is disabled. Please enable Ollama or use OpenAI backend."
            elif "Cannot connect to Ollama API" in error_message:
                error_message = "Cannot connect to Ollama API. Please make sure Ollama is running."
                
            logger.exception(f"Error processing query: {error_message}")
            self.write_message(json.dumps({
                "type": "query_result",
                "data": {
                    "status": "error",
                    "message": f"Error processing query: {error_message}"
                }
            }))
    
    def get_actual_files_from_directory(self, directory_path):
        """Get actual files from the specified directory"""
        try:
            files = []
            for file_path in directory_path.glob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ACCEPTED_EXTENSIONS:
                    file_stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": self.format_file_size(file_stat.st_size),
                        "created": datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d"),
                        "last_modified": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })
            return files
        except Exception as e:
            logger.exception(f"Error scanning directory {directory_path}: {str(e)}")
            return []
    
    def format_file_size(self, size_in_bytes):
        """Format file size in human-readable format"""
        if size_in_bytes < 1024:
            return f"{size_in_bytes} bytes"
        elif size_in_bytes < 1024 * 1024:
            return f"{size_in_bytes / 1024:.1f} KB"
        else:
            return f"{size_in_bytes / (1024 * 1024):.1f} MB"
    
    async def generate_csv_for_document(self, doc_id, filename, extracted_data):
        """Generate a CSV file with extracted data from the document analysis"""
        # Instead of implementing CSV generation here, use the agent_system
        logger.info(f"Generating CSV for document: {filename}")
        
        try:
            # Find the actual file path
            file_path = None
            possible_paths = [
                SAMPLES_DIR / filename,
                UPLOADS_DIR / filename
            ]
            
            for path in possible_paths:
                if path.exists():
                    file_path = str(path)
                    break
            
            if not file_path:
                logger.error(f"Cannot find source file for {filename}")
                # Fallback to a default CSV
                basename = os.path.splitext(filename)[0]
                csv_path = CSVS_DIR / f"{basename}_extracted.csv"
                
                # Create a basic CSV
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Entity", "Value", "Confidence"])
                    writer.writerow(["Error", "N/A", "Source file not found"])
                
                return str(csv_path)
            
            # Use the agent system to generate the CSV
            csv_result = await generate_csv(file_path, extracted_data)
            
            if csv_result["status"] == "success":
                # The csv_generator_agent should have created the CSV file
                # Get the CSV path from the result
                if "csv_path" in csv_result["result"]:
                    return csv_result["result"]["csv_path"]
                    
                # If no CSV path in result, construct default path
                basename = os.path.splitext(filename)[0]
                return str(CSVS_DIR / f"{basename}_extracted.csv")
            else:
                # Error occurred, create a fallback CSV
                logger.error(f"Error generating CSV: {csv_result.get('message')}")
                basename = os.path.splitext(filename)[0]
                csv_path = CSVS_DIR / f"{basename}_extracted.csv"
                
                # Create a basic CSV with error info
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Entity", "Value", "Confidence"])
                    writer.writerow(["Error", "N/A", csv_result.get("message", "Unknown error")])
                
                return str(csv_path)
                
        except Exception as e:
            logger.error(f"Error in generate_csv_for_document: {str(e)}")
            # Create a fallback CSV
            basename = os.path.splitext(filename)[0]
            csv_path = CSVS_DIR / f"{basename}_extracted.csv"
            
            # Create a basic CSV with error info
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Entity", "Value", "Confidence"])
                writer.writerow(["Error", "N/A", str(e)])
            
            return str(csv_path)
    
    async def save_metadata(self, documents):
        """Save document metadata to the metadata file"""
        # Check if metadata file exists
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, 'r') as f:
                    existing_metadata = json.load(f)
                existing_docs = existing_metadata.get("documents", [])
            except json.JSONDecodeError:
                existing_docs = []
        else:
            existing_docs = []
        
        # Process each document
        for doc in documents:
            # Set additional fields if missing
            if "report_period" not in doc:
                doc["report_period"] = "Unknown"
            if "client_name" not in doc:
                doc["client_name"] = "Unknown"
            if "account_name" not in doc:
                doc["account_name"] = "Unknown"
            if "wallet_id" not in doc:
                doc["wallet_id"] = f"WLT_{hash(doc.get('path', '')) % 10000:04d}"
            if "information_present" not in doc:
                doc["information_present"] = []
            
            # Generate CSV path if missing
            if "csv_path" not in doc:
                filename = os.path.basename(doc.get("path", "unknown.txt"))
                csv_filename = os.path.splitext(filename)[0] + "_extracted.csv"
                doc["csv_path"] = os.path.join("data/csvs", csv_filename)
            
            # Check if document already exists in metadata
            found = False
            for i, existing_doc in enumerate(existing_docs):
                if existing_doc.get("id") == doc.get("id"):
                    # Update existing document
                    existing_docs[i] = doc
                    found = True
                    break
            
            # Add new document if not found
            if not found:
                existing_docs.append(doc)
        
        # Save updated metadata
        metadata = {"documents": existing_docs}
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_session(self):
        """Save the current session data and notify clients"""
        try:
            # Send current session data to client via WebSocket
            message = {
                "type": "status_update",
                "data": self.session_data
            }
            self.write_message(json.dumps(message))
        except Exception as e:
            logger.error(f"Error in save_session: {str(e)}")

    def update_progress(self, progress_percent, status_message, file_name=None):
        """Update progress information and save session"""
        # Update progress information
        self.session_data["progress"]["current_file_progress"] = progress_percent
        self.session_data["progress"]["status_message"] = status_message
        
        # Update current file if provided
        if file_name:
            self.session_data["current_file"] = file_name
        
        # Save session to notify clients
        self.save_session()

def make_app():
    """Create the Tornado application"""
    static_path = os.path.join(Path(__file__).parent, "static")
    template_path = os.path.join(Path(__file__).parent, "templates")
    
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/api/status", StatusHandler),
        (r"/api/documents", DocumentsHandler),
        (r"/api/documents/([^/]+)", DocumentHandler),
        (r"/api/documents/([^/]+)/download", DocumentDownloadHandler),
        (r"/api/csv/(.*)", CSVHandler, {"path": str(CSVS_DIR)}),
        (r"/csv/download/(.*)", tornado.web.StaticFileHandler, {
            "path": str(CSVS_DIR),
            "default_filename": "data.csv"
        }),
        (r"/ws", WebSocketHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path}),
    ], 
    template_path=template_path,
    static_path=static_path,
    debug=options.debug
    )

async def initialize():
    """Initialize the application"""
    # Create required directories if they don't exist
    logger.info("Ensuring required directories exist")
    for directory in [SAMPLES_DIR, CSVS_DIR, UPLOADS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory {directory} is ready")
    
    try:
        # Initialize agent system
        logger.info("Initializing agent system...")
        result = await fin_assistant.agent_system.initialize_agents()
        
        # Check if the result is an integer (number of agents initialized)
        if isinstance(result, int):
            logger.info(f"Agent system initialized successfully with {result} agents")
            return True
        elif isinstance(result, dict) and result.get("status") == "success":
            logger.info("Agent system initialized successfully")
            return True
        else:
            # Handle invalid result format
            if not isinstance(result, dict):
                error_msg = f"Invalid result format from initialize_agents: {type(result)}"
                logger.critical(error_msg)
            else:
                error_msg = f"Failed to initialize agent system: {result.get('message', 'Unknown error')}"
                logger.critical(error_msg)
            
            logger.critical("The application requires a properly configured agent system.")
            logger.critical("You can still start the application, but document processing will be limited.")
            
            # Return True to allow starting with limited functionality
            return True
            
    except ImportError as e:
        # This is already handled in the import section, but just in case
        logger.critical(f"ImportError initializing agent system: {str(e)}")
        logger.critical("This application requires the OpenAI Agents SDK. Exiting.")
        return False
    except Exception as e:
        logger.critical(f"Error initializing agent system: {str(e)}")
        logger.critical("You can still start the application, but document processing will be limited.")
        
        # Return True to allow starting with limited functionality
        return True

def main():
    """Main entry point"""
    # Parse command line options
    tornado.options.parse_command_line()
    
    # Initialize the application
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if not loop.run_until_complete(initialize()):
        sys.exit(1)
    
    # Create and start the server
    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    
    logger.info("=== Financial Document Intelligence System ===")
    logger.info(f"Starting Tornado WebSocket server on port {options.port}")
    logger.info(f"Web interface available at: http://127.0.0.1:{options.port}")
    logger.info(f"WebSocket endpoint available at: ws://127.0.0.1:{options.port}/ws")
    
    # Start the IO loop
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main() 