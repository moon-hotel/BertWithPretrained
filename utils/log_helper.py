import logging
import os
from datetime import datetime

class Logger(object):

    def __init__(self, log_file_name, log_level, logger_name = "debug",log_dir = './logs/'):
        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)

        # 指定日志的最低输出级别，默认为WARN级别
        self.__logger.setLevel(log_level)

        # 指定路径
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 创建一个handler用于写入日志文件
        log_path = os.path.join(log_dir,log_file_name+'_'+str(datetime.now())[:10]+'.txt')
        file_handler = logging.FileHandler(log_path)

        # 创建一个handler用于输出控制台
        console_handler = logging.StreamHandler()

        # 定义handler的输出格式
        console_formatter = logging.Formatter('-- %(levelname)s: %(message)s')
        file_formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # 给logger添加handler
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
