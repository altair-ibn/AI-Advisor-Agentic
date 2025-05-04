#!/usr/bin/env python3
"""
Financial Document Intelligence System - Test Report Agents

This script tests the functionality of the specialized report type agents.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from fin_assistant
from fin_assistant import config
from fin_assistant.agent_system import initialize_agents
from fin_assistant.agent_system.runflow import analyze_file_type, process_report

async def test_report_agents():
    """Test the report type agents with sample financial documents"""
    logger.info("Initializing agents...")
    
    # Initialize agents
    agent_count = initialize_agents()
    logger.info(f"Successfully initialized {agent_count} agents")
    
    # Process sample documents
    samples_dir = config.DATA_DIR / "samples"
    
    if not samples_dir.exists():
        logger.error(f"Samples directory not found: {samples_dir}")
        return
    
    # Get sample files
    sample_files = list(samples_dir.glob("*"))
    if not sample_files:
        logger.warning(f"No sample files found in {samples_dir}")
        return
    
    logger.info(f"Found {len(sample_files)} sample files")
    
    # Process each sample file
    for file_path in sample_files:
        logger.info(f"Processing {file_path.name}...")
        
        try:
            # Analyze file type
            file_type_result = await analyze_file_type(str(file_path))
            
            if file_type_result["status"] == "success":
                logger.info(f"File type analysis successful: {file_type_result.get('document_type', 'Unknown')}")
                
                # Get report type
                document_type = file_type_result.get("document_type", "").lower()
                report_type = None
                
                # Map document type to report type
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
                
                # Process the report if a type was identified
                if report_type:
                    logger.info(f"Processing {report_type}...")
                    report_result = await process_report(str(file_path), report_type)
                    
                    if report_result["status"] == "success":
                        logger.info(f"Successfully processed {report_type}")
                        logger.info(f"Metadata: {json.dumps(report_result.get('metadata', {}), indent=2)}")
                        logger.info(f"CSV file: {report_result.get('csv_path')}")
                    else:
                        logger.error(f"Error processing {report_type}: {report_result.get('message')}")
                else:
                    logger.warning(f"Could not determine report type for {file_path.name}")
            else:
                logger.error(f"Error analyzing file type: {file_type_result.get('message')}")
        except Exception as e:
            logger.exception(f"Error processing {file_path.name}: {str(e)}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    asyncio.run(test_report_agents()) 