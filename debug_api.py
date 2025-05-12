#!/usr/bin/env python3
"""
Debug script to run the API server with full debug logging enabled.
This will help trace the error in the chat API endpoint.
"""

import sys
import os
import logging
import subprocess
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_debug_api():
    """Run the API server with debug enabled and log all output."""
    logger.info("Starting API server in debug mode")
    
    # Set environment variables for debugging
    env = os.environ.copy()
    env["DEBUG"] = "1"
    env["LOG_LEVEL"] = "DEBUG"
    
    # Kill any existing API server
    try:
        logger.info("Stopping any existing API server instances")
        subprocess.run(["pkill", "-f", "api.main"], check=False)
        time.sleep(1)  # Give time for the server to shut down
    except Exception as e:
        logger.warning(f"Error stopping existing API server: {e}")
    
    # Start the API server in a new process
    try:
        logger.info("Starting API server with debug logging")
        process = subprocess.Popen(
            ["python", "-m", "api.main"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        logger.info("Waiting for API server to start...")
        time.sleep(3)
        
        # Print information on how to test
        logger.info("\n" + "=" * 50)
        logger.info("API server started with debug logging")
        logger.info("=" * 50)
        logger.info("To test the chat API, run:")
        logger.info("./test_chat_api_endpoint.py")
        logger.info("=" * 50 + "\n")
        
        # Stream output from the server
        logger.info("Streaming server logs (press Ctrl+C to stop):")
        while True:
            # Read output line by line
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if stdout_line:
                print(f"[STDOUT] {stdout_line.strip()}")
            if stderr_line:
                print(f"[STDERR] {stderr_line.strip()}")
            
            # Check if process has exited
            if process.poll() is not None:
                break
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.error(f"Error running API server: {e}")
    finally:
        # Clean up
        try:
            if process and process.poll() is None:
                logger.info("Terminating API server process")
                process.terminate()
                process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error terminating API server: {e}")

if __name__ == "__main__":
    run_debug_api() 