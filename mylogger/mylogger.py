import logging
import os.path
import datetime


class mylogger:
    def __init__(self):
        path = self.getpath()
        logger = logging.getLogger('iCARL')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(path + ".log")
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logger = logger

    def getpath(self):
        today_date = datetime.date.today()
        i = 0
        path = './logger/' + str(today_date) + '/' + str(i)
        while True:
            if not os.path.exists(path):
                os.makedirs(path)
                break
            else:
                i += 1
                path = './logger/' + str(today_date) + '/' + str(i)

        return path + '/' + 'logger'


if __name__ == '__main__':
    mylogger = mylogger()
    mylogger.logger.info('hello') 
