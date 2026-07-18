import logging
import os
from logging.handlers import RotatingFileHandler

class LogFile:
    """
    A class to configure and instantiate a logger that writes to a file.
    """

    def __init__(
        self,
        log_file_path: str,
        log_name: str = __name__,
        log_level: int = logging.DEBUG,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        max_bytes: int = 10*1024*1024,  # 10 MB
        backup_count: int = 5
    ):
        """
        Initializes and configures the logger.

        Args:
            log_file_path (str): The full path to the log file.
            log_name (str): The name for the logger.
            log_level (int): The logging level (e.g., logging.INFO).
            log_format (str): The format for log messages.
            max_bytes (int): The maximum size of a log file before rotation.
            backup_count (int): The number of backup log files to keep.
        """
        self.log_file_path = log_file_path
        self.log_name = log_name
        self.log_level = log_level
        self.log_format = log_format
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        self._setup_logger()

    def _setup_logger(self):
        """
        Sets up the logger instance with handlers and formatters.
        """
        # Ensure the log directory exists
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Get logger instance
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(self.log_level)

        # Prevent adding handlers multiple times
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create a rotating file handler
        handler = RotatingFileHandler(
            self.log_file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        handler.setLevel(self.log_level)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter(self.log_format)
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: The configured logger instance.
        """
        return self.logger

if __name__ == '__main__':
    # Example usage:
    # Create a logger instance that writes to 'app.log' in a 'logs' directory
    log_file_instance = LogFile(log_file_path='logs/app.log', log_name='my_app')
    
    # Get the logger object
    logger = log_file_instance.get_logger()

    # Log messages
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    print(f"Log messages written to {os.path.abspath('logs/app.log')}")