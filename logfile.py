import time
from datetime import datetime,timedelta,date
def createLog(info):
    stamp=str(datetime.now()).split('.')
    timestamp=stamp[0].split(" ")
    day=str(date.today().strftime("%A"))
    filename=str(date.today())
    filename="log/"+filename+".txt"
    f=open(filename,"a+")
    f.write(timestamp[1]+" - "+ info+"\n")
    #f.write(info+"\n")
    f.close
