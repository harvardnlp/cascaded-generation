import sys, torch
import time

print ('start')

time.sleep(5)
a = torch.zeros(1, 4, 5)
b = a
a*b
