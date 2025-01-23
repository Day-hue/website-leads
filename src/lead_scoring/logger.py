import logging
import logging.config
import os
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler

# Constants for formatters
FORMATTER_JSON = 'json'
FORMATTER_DETAILED = 'detailed'

# Define the logfile name using the current date and time
log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Path to logfile (use a configurable environment variable for flexibility)
LOG_DIR = os.getenv('LOG_DIR', os.getcwd())
log_file_path = os.path.join(LOG_DIR, 'logs', log_file_name)

# Create a log directory if it does not exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Logger Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        },
        FORMATTER_DETAILED: {
            'format': '[%(asctime)s] %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
        },
        FORMATTER_JSON: {
            '()': jsonlogger.JsonFormatter,
            'format': '%(asctime)s %(name)s %(levelname)s %(module)s %(funcName)s %(lineno)d %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file_path,
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 20,
            'encoding': 'utf8',
            'formatter': FORMATTER_JSON  # JSON for the file
        },
        'console': {
            'level': 'DEBUG',  # Set console level to DEBUG for detailed output
            'class': 'logging.StreamHandler',
            'formatter': FORMATTER_DETAILED,  # Detailed format for the console
            'stream': sys.stdout,
        }
    },
    'loggers': {
        'LeadScoring': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Get the logger instance
logger = logging.getLogger('LeadScoring')

# # Example usage
# logger.debug("[DEBUG] Test debug message.")
# logger.info("[INFO] Test info message.")
# logger.warning("[WARNING] Test warning message.")
# logger.error("[ERROR] Test error message.")
# logger.critical("[CRITICAL] Test critical message.")
