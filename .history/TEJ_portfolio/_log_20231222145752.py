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
fmt = logging.Formatter('''%(asctime)s %(levelname)s %(message)s \
%(filename)s %(lineno)d %(name)s %(funcName)s \
%(process)d %(thread)d %(threadName)s %(processName)s \
                        %(relativeCreated)d %(msecs)d %(created)d %(asctime)s''')
handler.setFormatter(fmt)

# 5. 將handler加入logger
logger.addHandler(handler)

def func():
    try:
        n = int(input('請輸入數字:'))
        for i in range(n):
            print(i)
    except:
        logger.error('wrong type')
    finally:
        print('程式結束')

func()