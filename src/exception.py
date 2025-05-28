import sys  # Gives access to system-specific parameters and functions
from src.logger import logging  # Custom logging setup from your project

# Function to format and return detailed error messages
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get exception traceback information
    file_name = exc_tb.tb_frame.f_code.co_filename  # Extract the filename where error occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Format the error with filename, line, and message
    )
    return error_message  # Return the full custom error message

# Custom exception class to override the default error behavior
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Call the base class constructor
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Create detailed message

    def __str__(self):
        return self.error_message  # Return the custom error message when the exception is printed

    
