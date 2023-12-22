import logging

# 1. 日誌對象
logger = logging.getLogger()

# 2. 設置級別
logger.setLevel(logging.DEBUG)

# 3. 創建handler
file = 't2/log.txt'
handler = logging.FileHandler(file)
handler.setLevel(logging.DEBUG)

# 4. 創建formatter
fmt = logging.Formatter('''
                        %(asctime)s %(levelname)s %(message)s \
                        %(filename)s %(lineno)d %(name)s %(funcName)s \
                        %(process)d %(thread)d %(threadName)s %(processName)s \
                        %(relativeCreated)d %(msecs)d %(created)d %(asctime)
                        ''')
handler.setFormatter(fmt)

# 5 
