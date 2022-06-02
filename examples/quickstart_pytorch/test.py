import os
import sys

if __name__ == '__main__':
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    curPath = os.path.abspath(os.path.dirname(__file__))
    print(curPath)
    rootPath = os.path.split(curPath)[0]
    print(rootPath)
    sys.path.append(rootPath)
