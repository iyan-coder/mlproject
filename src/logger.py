import logging         # Built-in module to handle logging of messages
import os              # Module to interact with the operating system
from datetime import datetime  # To generate timestamps for log filenames

# Create a log filename using the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path where logs will be saved, under a 'logs' folder
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the 'logs' directory if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file that will be written to
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up the logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Log messages will be saved to this file
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Format of each log message
    level=logging.INFO  # Set the logging level to INFO (can also be DEBUG, WARNING, ERROR, etc.)
)



