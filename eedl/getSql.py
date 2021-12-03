import csv


class SqlInterface:
    sql = []

    def __init__(self):
        f = csv.reader(open('train.csv', 'r'))
        for i in f:
            if len(i) > 0:
                self.sql.append(i[0])

    def get(self, i):
        if i < len(self.sql):
            return self.sql[i]
        return None
