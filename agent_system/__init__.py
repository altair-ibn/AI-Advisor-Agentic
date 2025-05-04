"""
Financial Document Intelligence System - Agent Module

This module provides the agent framework for the application.
"""

import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY environment variable not set. Agents may not function properly.")

# Import to avoid circular dependencies
from .tools import (
    register_directory_tools,
    register_document_tools,
    register_csv_tools,
    register_ollama_tools,
    register_metadata_tools,
    register_query_tools,
    register_file_analyzer_tools,
)

from .agent_definitions import (
    directory_scanner_agent,
    file_analyzer_agent,
    document_analyzer_agent,
    metadata_agent,
    csv_generator_agent,
    query_agent,
    annual_report_agent,
    audit_report_agent,
    balance_sheet_agent,
    income_statement_agent,
    cash_flow_statement_agent,
    tax_document_agent
)

from .runflow import (
    run_directory_scan,
    process_document,
    process_query,
    analyze_file_type,
    process_report,
    generate_csv,
)

async def initialize_agents() -> int:
    """
    Initialize the agent system
    
    Returns:
        Number of agents initialized
    """
    # Register tools with agents
    register_directory_tools(directory_scanner_agent)
    register_file_analyzer_tools(file_analyzer_agent)
    register_document_tools(document_analyzer_agent)
    register_metadata_tools(metadata_agent)
    register_csv_tools(csv_generator_agent)
    register_ollama_tools(file_analyzer_agent)
    register_ollama_tools(document_analyzer_agent)
    register_ollama_tools(annual_report_agent)
    register_ollama_tools(audit_report_agent)
    register_ollama_tools(balance_sheet_agent)
    register_ollama_tools(income_statement_agent)
    register_ollama_tools(cash_flow_statement_agent)
    register_ollama_tools(tax_document_agent)
    register_ollama_tools(metadata_agent)
    register_ollama_tools(csv_generator_agent)
    register_query_tools(query_agent)
    
    # Number of agents
    agent_count = 12
    
    logger.info(f"Successfully initialized {agent_count} agents")
    
    return agent_count 