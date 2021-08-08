import logging, os, sys
from logging.handlers import RotatingFileHandler


def get_logger(name, project_home):
    """
    Args:
        name(str):생성할 log 파일명입니다.

    Returns:
         생성된 logger객체를 반환합니다.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    rotate_handler = RotatingFileHandler(
        os.path.join(project_home, "out", "logs", name + ".log"),
        'a',
        1024 * 1024 * 2,
        5
    )
    # formatter = logging.Formatter(
    #     '[%(levelname)s]-%(asctime)s-%(filename)s:%(lineno)s:%(message)s',
    #     datefmt="%Y-%m-%d %H:%M:%S"
    # )

    formatter = logging.Formatter(
        ''
    )

    rotate_handler.setFormatter(formatter)
    logger.addHandler(rotate_handler)
    return logger