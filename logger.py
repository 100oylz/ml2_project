import logging
import os.path

from colorlog import ColoredFormatter


def logConfig(path_format: str, task_format, add_terminal: False, *data):
    formatter = ColoredFormatter(
        "%(white)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'white',
            'WARNING': 'white',
            'ERROR': 'white',
            'CRITICAL': 'white,bg_red',
        },
        secondary_log_colors={},
        style='%'
    )
    task_name = task_format.format(*data)
    logger = logging.getLogger(f'{task_name}_logger')
    logger.setLevel(logging.DEBUG)

    log_filename = os.path.join(path_format, f'{task_name}.log')
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    if (add_terminal):
        # 创建一个用于在控制台输出的处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
