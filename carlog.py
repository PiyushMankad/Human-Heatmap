import time
import datetime
def createLog(info):
    stamp=str(datetime.datetime.now()).split('.')
    timestamp=stamp[0].split(" ")
    day=str(datetime.date.today().strftime("%A"))
    f=open('log.txt',"a+")
    f.write(timestamp[0]+': '+day+': '+timestamp[1]+" -"+ info+"\n")
    f.close

