import psycopg2


class DBInterface:
    conn = psycopg2.connect(database="DataTable", user="postgres",
                            password="", host="localhost", port="5432")
    cursor = conn.cursor()

    def __init__(self, table_name):
        self.cursor.execute("select * from " + table_name + " ;")
        self.table_row_num = len(self.cursor.fetchall())

    def getTrueCard(self, sql):
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return len(rows)

    def getEsCard(self, sql):
        self.cursor.execute("explain " + sql)
        rows = self.cursor.fetchall()
        str1 = rows[0]
        strs = str1[0].split("rows=")
        s1 = strs[1].split(" ")
        return s1[0]
