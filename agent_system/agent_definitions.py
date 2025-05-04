"""
Financial Document Intelligence System - Agent Definitions

This module defines the agents used in the system.
"""

import logging
from typing import Dict, Any, List, Optional

from agents import Agent
try:
    from fin_assistant import config
except ImportError:
    import sys
    import os
    # Try to add parent directory to path and import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from fin_assistant import config

logger = logging.getLogger(__name__)

# Define the Directory Scanner Agent
directory_scanner_agent = Agent(
    name="DirectoryScanner",
    instructions="""
    You are a Directory Scanner Agent. Your job is to scan directories for financial documents.
    When scanning a directory, report all files found along with their details.
    """
)

# Define the File Analyzer Agent
file_analyzer_agent = Agent(
    name="FileAnalyzer",
    instructions="""
    You are a File Analyzer Agent. Your job is to analyze files to determine their type, 
    such as balance sheet, income statement, cash flow statement, etc.
    Use the content and file name to make accurate determinations.
    """
)

# Define the Document Analyzer Agent
document_analyzer_agent = Agent(
    name="DocumentAnalyzer",
    instructions="""
    You are a Document Analyzer Agent. Your job is to analyze financial documents and extract key information.
    Focus on finding financial metrics, entities, dates, and important information.
    Ensure your analysis is thorough and captures all relevant financial data.
    """
)

# Define specialized report type agents

# Annual Report Agent
annual_report_agent = Agent(
    name="AnnualReportAgent",
    instructions="""
    You are an Annual Report Agent specializing in analyzing annual reports.
    Extract key information like revenue, profit, performance metrics, and outlooks.
    Your analysis should cover all major sections of an annual report.
    """
)

# Audit Report Agent
audit_report_agent = Agent(
    name="AuditReportAgent",
    instructions="""
    You are an Audit Report Agent specializing in analyzing audit reports.
    Extract key findings, recommendations, compliance issues, and risk assessments.
    Focus on delivering a comprehensive analysis of audit findings.
    """
)

# Balance Sheet Agent
balance_sheet_agent = Agent(
    name="BalanceSheetAgent",
    instructions="""
    You are a Balance Sheet Agent specializing in analyzing balance sheets.
    Extract asset, liability, and equity information with precise values.
    Ensure you capture all line items and their values accurately.
    """
)

# Income Statement Agent
income_statement_agent = Agent(
    name="IncomeStatementAgent",
    instructions="""
    You are an Income Statement Agent specializing in analyzing income statements.
    Extract revenue, expense, and profit information with precise values.
    Focus on capturing all line items and their financial values accurately.
    """
)

# Cash Flow Statement Agent
cash_flow_statement_agent = Agent(
    name="CashFlowStatementAgent",
    instructions="""
    You are a Cash Flow Statement Agent specializing in analyzing cash flow statements.
    Extract operating, investing, and financing activities with precise values.
    Ensure you capture all cash flow items and their values accurately.
    """
)

# Tax Document Agent
tax_document_agent = Agent(
    name="TaxDocumentAgent",
    instructions="""
    You are a Tax Document Agent specializing in analyzing tax documents.
    Extract tax liabilities, credits, deductions, and other tax-related information.
    Focus on capturing all tax-related items and their values accurately.
    """
)

# Define the Metadata Agent
metadata_agent = Agent(
    name="MetadataAgent",
    instructions="""
    You are a Metadata Agent. Your job is to create metadata for financial documents.
    Extract information like report type, period, client name, and description.
    Ensure all metadata fields are complete and accurate.
    """
)

# Define the CSV Generator Agent
csv_generator_agent = Agent(
    name="CSVGenerator",
    instructions="""
    You are a CSV Generator Agent. Your job is to generate structured CSV files from financial document analysis.
    Extract all financial data from the document, including tables, metrics, and values.
    Ensure the CSV is well-structured with appropriate columns and complete data.
    Always include the actual financial data from the source document, not just metadata.
    """
)

# Define the Query Agent
query_agent = Agent(
    name="QueryAgent",
    instructions="""
    You are a Query Agent. Your job is to answer questions about financial documents.
    Use the available data to provide accurate answers.
    When responding, cite the specific document and data points you used.
    """
)

# Define the Data Extraction Agent
data_extraction_agent = Agent(
    name="DataExtractionAgent",
    instructions="""
    You are a Financial Data Extraction Agent specialized in extracting structured data from financial documents.
    Your primary responsibility is to:
    
    1. Analyze the source document to identify its type (balance sheet, income statement, cash flow, etc.)
    2. Extract all financial figures, line items, and their corresponding values
    3. Preserve the hierarchical structure of the financial data
    4. Format the data into a structured JSON format suitable for CSV conversion
    5. Ensure all numerical values are properly extracted with correct units
    6. Maintain relationships between items (e.g., subtotals, totals)
    
    When extracting data, focus on:
    - Accurate identification of financial line items
    - Precise extraction of monetary values
    - Proper categorization of data points
    - Maintaining the semantic structure of the document
    
    Even with incomplete or poorly formatted documents, extract as much valid financial data as possible.
    Be comprehensive in your extraction, capturing all financial figures present in the document.
    
    Format the extracted data following the structure:
    {
      "columns": ["Item", "Value", "Category", "Period", "Notes"],
      "data": [
        {"Item": "Revenue", "Value": "100000", "Category": "Income", "Period": "2023", "Notes": "Annual revenue"},
        ... (more rows)
      ]
    }
    """
)

# Set up handoffs between agents
directory_scanner_agent.handoffs = [file_analyzer_agent]
file_analyzer_agent.handoffs = [
    annual_report_agent, 
    audit_report_agent, 
    balance_sheet_agent, 
    income_statement_agent,
    cash_flow_statement_agent,
    tax_document_agent,
    document_analyzer_agent  # fallback if no specific type is identified
]

# Set up handoffs from specialized report agents to metadata and CSV generator
annual_report_agent.handoffs = [metadata_agent, csv_generator_agent]
audit_report_agent.handoffs = [metadata_agent, csv_generator_agent]
balance_sheet_agent.handoffs = [metadata_agent, csv_generator_agent]
income_statement_agent.handoffs = [metadata_agent, csv_generator_agent]
cash_flow_statement_agent.handoffs = [metadata_agent, csv_generator_agent]
tax_document_agent.handoffs = [metadata_agent, csv_generator_agent]

# Other handoffs
document_analyzer_agent.handoffs = [metadata_agent, csv_generator_agent]
metadata_agent.handoffs = [csv_generator_agent]
csv_generator_agent.handoffs = [query_agent]
query_agent.handoffs = [document_analyzer_agent, csv_generator_agent]

# Don't register tools yet - we'll do that in separate modules

def get_all_agents() -> Dict[str, Agent]:
    """
    Get all defined agents
    
    Returns:
        Dict mapping agent names to Agent instances
    """
    return {
        "directory_scanner": directory_scanner_agent,
        "file_analyzer": file_analyzer_agent,
        "document_analyzer": document_analyzer_agent,
        "annual_report": annual_report_agent,
        "audit_report": audit_report_agent,
        "balance_sheet": balance_sheet_agent,
        "income_statement": income_statement_agent,
        "cash_flow_statement": cash_flow_statement_agent,
        "tax_document": tax_document_agent,
        "metadata": metadata_agent,
        "csv_generator": csv_generator_agent,
        "query_agent": query_agent,
        "data_extraction": data_extraction_agent
    } 