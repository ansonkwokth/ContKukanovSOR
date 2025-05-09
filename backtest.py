from utils.logger import get_logger

logger = get_logger("MyModule")

logger.debug("This is a debug message.")
logger.info("Starting the process.")
logger.warning("Something might be wrong.")
logger.error("An error occurred.")
logger.critical("System failure.")

