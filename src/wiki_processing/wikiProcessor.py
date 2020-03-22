import json
import re
import os
import mmh3
import sys
import random


def walk(basePath):
    filePaths = []
    base = os.listdir(basePath)
    for dr in base:
        for file in os.listdir(basePath + "/" + dr):
            filePaths.append(basePath + "/" + dr + "/" + file)
    return filePaths


def hashParagraph(words, bound):
    hashes = {}
    for word in words:
        h = mmh3.hash(word) % bound
        hashes[h] = hashes.get(h, 0) + 1
    return hashes


def readWiki(path):
    pSplit = re.compile("\\n+")
    wSplit = re.compile("\s")
    numArticles = 0
    numParagraphs = 0
    longest = 0
    shortest = float('inf')
    with open(path, encoding="UTF-8") as jsonFile:
        for line in jsonFile.readlines():
            numArticles += 1
            data = json.loads(line)
            text = data["text"]
            paragraphs = re.split(pSplit, text)
            for p in paragraphs:
                words = re.split(wSplit, p)
                if (len(words) < 10):
                    continue
                numParagraphs += 1
                l = len(words)
                longest = max(longest, l)
                shortest = min(shortest, l)
    return numArticles, numParagraphs, longest, shortest


def convertToLine(docId, hashes):
    line = str(docId)
    keys = sorted(list(hashes.keys()))
    for key in keys:
        line += " " + str(key) + ":" + str(hashes[key])
    return line + "\n", len(hashes)


def hashWiki(path, bound):
    pSplit = re.compile("\\n+")
    wSplit = re.compile("\s")
    docHashes = []
    maxNonZero = 0
    with open(path, encoding="UTF-8") as jsonFile:
        for line in jsonFile.readlines():
            data = json.loads(line)
            docId = data["id"]
            text = data["text"]
            paragraphs = re.split(pSplit, text)
            for p in paragraphs:
                words = re.split(wSplit, p)
                if (len(words) < 10):
                    continue
                paragraphHashes = hashParagraph(words, bound)
                hashLine, nonZero = convertToLine(docId, paragraphHashes)
                docHashes.append(hashLine)
                maxNonZero = max(maxNonZero, nonZero)
    return docHashes, maxNonZero


def scan(base):
    paths = walk(base)
    longest = 0
    shortest = float('inf')
    numParagraphs = 0
    numArticles = 0
    for path in paths:
        na, np, l, s = readWiki(path)
        longest = max(longest, l)
        shortest = min(shortest, s)
        numArticles += na
        numParagraphs += np
    print("NumArticles: " + str(numArticles) + " NumParagraphs: " + str(numParagraphs) +
          " LongestParagraph: " + str(longest) + " ShortestParagraph: " + str(shortest))


def hashAllFiles(base, outputFile, bound):
    lines = 0
    maxNonZero = 0
    paths = walk(base)
    with open(outputFile, 'w') as output:
        for path in paths:
            docHashes, nonZero = hashWiki(path, bound)
            maxNonZero = max(maxNonZero, nonZero)
            for line in docHashes:
                lines += 1
                output.write(line)
    print("Paragraphs: " + str(lines) + " MaxNonZero: " + str(maxNonZero))


def getParagraphs(base, outputFile):
    pSplit = re.compile("\\n+")
    lines = 0
    paths = walk(base)
    with open(outputFile, 'w', encoding="UTF-8") as output:
        for path in paths:
            with open(path, encoding="UTF-8") as jsonFile:
                for line in jsonFile.readlines():
                    data = json.loads(line)
                    docId = data["id"]
                    text = data["text"]
                    paragraphs = re.split(pSplit, text)
                    for p in paragraphs:
                        if len(p) > 50:
                            lines += 1
                            x = docId + ":" + p + "\n"
                            output.write(x)
    print("Paragraphs: " + str(lines))


def shuffleDataset(filename):
    file = open(filename, encoding="UTF-8")
    lines = file.readlines()
    print("Lines: " + str(len(lines)))
    random.shuffle(lines)
    file.close()
    file = open(filename, 'w', encoding="UTF-8")
    file.writelines(lines)
    file.close()
    print("Dataset Shuffled")

    # scan("extract")
# hashAllFiles("extract", "wiki", 2**20)


getParagraphs("extract", "wiki_paragraphs")

shuffleDataset("wiki_paragraphs")
