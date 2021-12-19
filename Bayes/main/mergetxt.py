import os

if __name__ == "__main__":
    filenames = os.listdir("../dictionary")
    file = open("initVocabulary.txt", mode="w", encoding="utf8")
    for filename in filenames:
        print(filename)
        lines = open("../dictionary/{filename}".format(filename=filename), encoding="utf8").readlines()
        # separator = "********{filename}********\n".format(filename=filename)
        # file.write(separator)
        file.writelines(lines)
        file.flush()