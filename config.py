"""
Financial Document Intelligence System - Configuration

This module provides configuration settings for the application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CSV_DIR = DATA_DIR / "csvs"
METADATA_PATH = DATA_DIR / "metadata.json"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_API = "http://localhost:11434"
OLLAMA_MODEL = "llama2"

# Processing configuration
MAX_TOKENS = 2048
CHUNK_OVERLAP = 200

# File extensions to process
ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".csv", ".txt", ".json"]

# Web server configuration
SERVER_PORT = 8080
DEBUG_MODE = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "app.log"

# OpenAI API configuration (for Agents SDK)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Agent models
DIRECTORY_SCANNER_MODEL = "gpt-3.5-turbo"
FILE_ANALYZER_MODEL = "gpt-4o"
DOCUMENT_ANALYZER_MODEL = "gpt-4o"
REPORT_AGENT_MODEL = "gpt-4o"
METADATA_AGENT_MODEL = "gpt-3.5-turbo"
CSV_GENERATOR_MODEL = "gpt-4o"
QUERY_AGENT_MODEL = "gpt-4o"

# Report type schemas
REPORT_SCHEMAS = {
    "annual_report": {
        "columns": [
            "company_name", "ticker", "fiscal_year", "report_date", 
            "key_metrics", "financial_highlights", "revenue", "net_income", 
            "total_assets", "total_liabilities", "total_equity", "earnings_per_share",
            "fiscal_year_end", "auditor", "risk_factors"
        ]
    },
    "audit_report": {
        "columns": [
            "company_name", "report_date", "auditor", "audit_opinion_type",
            "key_audit_matters", "material_weaknesses", "basis_for_opinion",
            "report_date_range", "auditing_standard"
        ]
    },
    "balance_sheet": {
        "columns": [
            "company_name", "report_date", "total_assets", "current_assets", 
            "non_current_assets", "total_liabilities", "current_liabilities", 
            "non_current_liabilities", "total_equity", "common_stock", 
            "retained_earnings", "reporting_currency", "fiscal_period"
        ]
    },
    "income_statement": {
        "columns": [
            "company_name", "report_period", "total_revenue", "cost_of_revenue",
            "gross_profit", "operating_expenses", "operating_income", 
            "other_income_expense", "income_before_tax", "income_tax_expense",
            "net_income", "earnings_per_share", "weighted_avg_shares",
            "reporting_currency", "fiscal_period"
        ]
    },
    "cash_flow_statement": {
        "columns": [
            "company_name", "report_period", "net_income", "operating_activities",
            "investing_activities", "financing_activities", "net_change_in_cash",
            "beginning_cash_balance", "ending_cash_balance", "reporting_currency",
            "fiscal_period"
        ]
    },
    "tax_document": {
        "columns": [
            "entity_name", "tax_id", "tax_period", "taxable_income", "tax_liability",
            "tax_credits", "tax_deductions", "tax_payments", "jurisdiction",
            "filing_status", "document_type"
        ]
    }
} 