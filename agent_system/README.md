# Financial Document Intelligence System - Agent System

This module provides a specialized agent system for analyzing and extracting data from financial documents. The system uses the OpenAI Agents SDK to create a hierarchy of specialized agents that work together to process different types of financial reports.

## Agent Structure

The agent system is organized as follows:

```
Directory Scanner Agent
    └── File Analyzer Agent
        ├── Annual Report Agent
        │   └── Metadata Agent
        │       └── CSV Generator Agent
        ├── Audit Report Agent
        │   └── Metadata Agent
        │       └── CSV Generator Agent
        ├── Balance Sheet Agent
        │   └── Metadata Agent
        │       └── CSV Generator Agent
        ├── Income Statement Agent
        │   └── Metadata Agent
        │       └── CSV Generator Agent
        ├── Cash Flow Statement Agent
        │   └── Metadata Agent
        │       └── CSV Generator Agent
        ├── Tax Document Agent
        │   └── Metadata Agent
        │       └── CSV Generator Agent
        └── Document Analyzer Agent (fallback)
            └── Metadata Agent
                └── CSV Generator Agent
```

## Agent Descriptions

1. **Directory Scanner Agent**: Scans directories for financial documents and organizes them for processing.

2. **File Analyzer Agent**: Analyzes uploaded files to determine the type of financial report (annual report, audit report, balance sheet, etc.) and hands off to the appropriate specialized agent.

3. **Report Type Agents**: Specialized agents for different report types:
   - **Annual Report Agent**: Extracts structured data from annual reports
   - **Audit Report Agent**: Extracts structured data from audit reports
   - **Balance Sheet Agent**: Extracts structured data from balance sheets
   - **Income Statement Agent**: Extracts structured data from income statements
   - **Cash Flow Statement Agent**: Extracts structured data from cash flow statements
   - **Tax Document Agent**: Extracts structured data from tax documents

4. **Metadata Agent**: Extracts and organizes metadata from financial documents.

5. **CSV Generator Agent**: Creates structured CSV files from the extracted data for analysis.

6. **Query Agent**: Answers queries about the processed financial documents.

## Integration with Ollama

Each specialized report agent uses Ollama to extract data for each column/section of the report. The workflow is:

1. The File Analyzer Agent identifies the report type
2. The appropriate Report Type Agent is called
3. The Report Agent uses Ollama to extract structured data for each column in the report
4. The Metadata Agent extracts and organizes document metadata
5. The CSV Generator Agent creates a structured CSV file from the data

## Processing Workflow

The main processing workflow is:

1. A file is uploaded or selected from a directory
2. The File Analyzer Agent determines the report type
3. The file is passed to the appropriate specialized report agent
4. The report agent extracts structured data using Ollama
5. Metadata is extracted and organized
6. A CSV file is generated for analysis
7. The system is ready to answer queries about the document

## Tools

Each agent has specific tools registered with it:

- **Directory Scanner Tools**: Tools for scanning directories and identifying files
- **File Analyzer Tools**: Tools for determining report types
- **Report Type Tools**: Specialized tools for extracting data from different report types
- **Metadata Tools**: Tools for extracting and organizing metadata
- **CSV Generator Tools**: Tools for creating structured CSV files
- **Query Tools**: Tools for answering queries about the documents

## Configuration

Report schemas and agent models are configured in `config.py`. You can customize:

- Report type schemas and columns
- Agent models (which OpenAI model to use for each agent)
- Ollama model configuration

## Testing

You can test the system using the `test_report_agents.py` script, which will process sample files in the `data/samples` directory and show the extracted data, metadata, and generated CSV files. 