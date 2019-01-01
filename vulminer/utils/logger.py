# -*- coding: utf-8 -*-
"""
logger.py - Provide the log ability

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import pathlib
import logging

LEVEL = {
    'notest': logging.NOTSET,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


class Logger:
    """
    Logger is the wrapper of logging.Logger, you can easily set level and
    formatter for logger and also just use it as logging.Logger. By default,
    logger only add the command line handler. Yon can add additional file
    handler by logger.addFileHandler.

    """

    def __init__(self):
        self._level = logging.INFO
        self._formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s')
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(self._level)

    def __getattr__(self, methods):
        return getattr(self._logger, methods)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, lv):
        if lv in LEVEL.keys():
            self._level = LEVEL[lv]
            self._logger.setLevel(self._level)
        else:
            raise ValueError(f'{lv} is not a correct logging level')

    @property
    def formatter(self):
        return self._formatter

    @formatter.setter
    def formatter(self, fmt):
        if isinstance(fmt, logging.Formatter):
            self._formatter = fmt
        else:
            raise ValueError(f'{fmt} is not a Formatter object')

    def addCmdHandler(self):
        """add command line handler for logger"""
        cmd_handler = logging.StreamHandler()
        cmd_handler.setLevel(self._level)
        cmd_handler.setFormatter(self._formatter)
        self._logger.addHandler(cmd_handler)

    def addFileHandler(self, path='vulminer.log'):
        """add file handler for logger

        Args:
            path <str>: The path of log file

        """
        path = pathlib.Path(path).expanduser()
        print(str(path))
        if not path.exists():
            path.touch()
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(self._level)
        file_handler.setFormatter(self._formatter)
        self._logger.addHandler(file_handler)


logger = Logger()
