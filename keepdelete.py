import time
import glob
import os

while True:
    filenames = glob.glob('checkpoints_debug/checkpoint*pt')
    for filename in filenames:
        if '_' not in os.path.basename(filename):
            print (filename)
            os.remove(filename)
    time.sleep(60)
    print ('-')
