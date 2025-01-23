import sys
from typing import Any, Tuple, Union
import logging

# Initialize the logger
logger = logging.getLogger('LeadScoring')
logging.basicConfig(level=logging.ERROR)

# Utility function to extract error message details
def error_message_detail(error: Union[Exception, str], error_details: Tuple[type, BaseException, type]) -> str:
    """
    Extract detailed error information including the file name and line number.

    Args:
        error (Exception | str): The raised exception or error message.
        error_details (tuple): The traceback information from sys.exc_info().

    Returns:
        str: A formatted error message.
    """
    try:
        # Get traceback information
        _, _, exc_tb = error_details

        # Extract the file name and line number
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        # Create a formatted error message
        formatted_error_message = (
            f"File: {file_name} \n"
            f"Line Number: [{line_number}] \n"
            f"Error message: [{error}]"
        )
        return formatted_error_message
    except AttributeError as attr_err:
        # Handle cases where traceback is unavailable
        return f"Error: {error} (Details unavailable: {attr_err})"

# Define the custom exception class
class CustomException(Exception):
    """
    Custom exception class for the Lead Scoring application.

    Attributes:
        error (Exception): The original exception instance.
        error_details (tuple): Traceback information from sys.exc_info().
    """
    def __init__(self, error: Union[Exception, str], error_details: Tuple[type, BaseException, type]):
        # Extract the error message details
        formatted_error_message = error_message_detail(error, error_details)

        # Log the error message
        logger.error(formatted_error_message)

        # Call the parent constructor with the formatted error message
        super().__init__(formatted_error_message)

        # Store the error and details object
        self.error = error
        self.error_details = error_details

    def __str__(self) -> str:
        """
        Override the __str__ method to return the formatted error message.

        Returns:
            str: The detailed error message.
        """
        return str(self.args[0])  # Use the message set in super().__init__()

def example_function():
    try:
        # Simulating an error
        1 / 0
    except Exception as e:
        raise CustomException(e, sys.exc_info())

if __name__ == "__main__":
    try:
        logger.info("Starting the example function.")
        example_function()
    except CustomException as e:
        logger.error(f"An error occurred: {e}")
