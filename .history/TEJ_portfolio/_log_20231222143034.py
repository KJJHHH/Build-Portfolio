import logging

# 1. 日誌對象
logger = logging.getLogger()

# 2. 設置級別
logger.setLevel(logging.DEBUG)

# 3. 創建handler
file = 't2/log.txt'
handlerlogging.FileHandler()

# 3. 設置輸出格式
