#!/usr/bin/env python3
"""
Financial Document Intelligence System - Main Entry Point

This is the main entry point for the application. It initializes the agent system
and runs the Tornado application.
"""

import os
import logging
import asyncio
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure we're in the project root for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize the agent system
async def init_agent_system():
    try:
        from fin_assistant.agent_system import initialize_agents
        result = await initialize_agents()
        logger.info(f"Agent system initialization: {result}")
        return result
    except ImportError:
        logger.warning("Agent system could not be imported. Running in demo mode.")
        return {"status": "error", "message": "Agent system not available"}
    except Exception as e:
        logger.exception(f"Error initializing agent system: {e}")
        return {"status": "error", "message": str(e)}

async def main_async():
    # Initialize agent system
    await init_agent_system()
    
    # Import the tornado app after agent system initialization
    from fin_assistant.tornado_app import main
    main()

def main():
    # Use asyncio.run() instead of manually creating and running a loop
    asyncio.run(init_agent_system())
    
    # Import and run the tornado app
    from fin_assistant.tornado_app import main as tornado_main
    tornado_main()

if __name__ == "__main__":
    main() 