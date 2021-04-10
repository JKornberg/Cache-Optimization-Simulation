from queue import Queue

class Cache:
    def __init__(self):
        x = 10
        y = 5
    def test(self):
        print(x)

q = Queue()
c = Cache()
q.push(c,c.test)

y = q.pop()