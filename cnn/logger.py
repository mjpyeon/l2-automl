import logging
import os
import pathlib
import pdb

LOG_LEVEL_DICT = {
    "logging.debug": logging.DEBUG,
    "logging.info": logging.INFO,
    "logging.warning": logging.WARNING,
    "logging.error": logging.ERROR,
    "logging.critical": logging.CRITICAL
}

DEFAULT_LOG_FORMAT = '[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s'
DEFAULT_LOG_LEVEL = LOG_LEVEL_DICT['logging.DEBUG'.lower()]

def create_logger(logger_name, log_file_path, log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT):
	"""
	Gets a logger object.
	
	:param logger_name: The name of the logger
	:param log_file_path: The path to write logs to
	:param log_level: The lowest log level to track
	:param log_format: A string providing the format for the logs
	:returns: Returns a thread-safe logging object.
	
	Using the defaut log format, the logger will include:
	- Timestamp
	- Filename of the file making the call
	- Line number within that file
	- Severity level of the log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
	- The log message
	in each log message, which are written to the file provided by log_file_path.
	Note that the logger will always append to this file and not overwrite it.
	
	An example would be:
	In foo.py:
	from .utils import logger
	logger = logger.get_logger('foo', '/var/log/foo.log')
	logger.warning('This is a warning message.')
	
	Then, in /var/log/foo.log, you would find
	[2018-06-26 15:24:41,037:foo.py#L3:WARNING]: This is a warning message.
	"""
	# for python 2 compatibility
	if(not pathlib.Path(log_file_path).parents[0].exists()):
		pathlib.Path(log_file_path).parents[0].mkdir()
	logger = logging.getLogger(logger_name)
	logger.setLevel(log_level)
	fh = logging.FileHandler(log_file_path)
	fh.setLevel(log_level)
	formatter = logging.Formatter(log_format)
	fh.setFormatter(formatter)
	sh = logging.StreamHandler()
	sh.setLevel(log_level)
	sh.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(sh)
	return logger

def get_logger(logger_name):
	return logging.getLogger(logger_name)