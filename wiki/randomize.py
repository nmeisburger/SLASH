import sys
import random

args = sys.argv
print(args[1])

file = open(args[1])
lines = file.readlines()
print("Lines: " + str(len(lines)))
random.shuffle(lines)
file.close()
file = open(args[1], 'w')
file.writelines(lines)
file.close()

print("Dataset Shuffled")


def shuffleDataset(filename):
    file = open(filename)
    lines = file.readlines()
    print("Lines: " + str(len(lines)))
    random.shuffle(lines)
    file.close()
    file = open(filename, 'w')
    file.writelines(lines)
    file.close()
    print("Dataset Shuffled")
