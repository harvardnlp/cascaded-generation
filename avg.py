import sys

tot_score = 0.
tot = 0
with open(sys.argv[1]) as fin:
    for line in fin:
        items = line.strip().split()
        score = sum([float(item) for item in items])
        tot_score += score
        tot += 1


print (tot_score / tot)
