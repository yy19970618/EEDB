import itertools
import numpy

class SlideBar(object):
    '''
    初始化滑动窗口为空
    '''
    def __init__(self):
        self.delete = {}
        self.insert = {}
        self.total = {}
        self.sortlist = []
        self.gamma = [0.8, 0.85, 0.9]  # 被删除查询的折旧因子
        
        self.index = [0, 0, 0]

        self.sortindex = [[],[],[]]  # 将不同个数的索引分开排序

        self.count = 0


    def InsertOne(self, iter, flag):

        for k in iter:
            # 插入项集
            self.insert[k] = self.insert.setdefault(k, 0) + 1
            self.delete[k] = self.delete.setdefault(k, 0)
            id = self.total.setdefault(k, -1)
            # 是一个新项集
            if id == -1:
                self.total[k] = self.index[flag] # 将排序下标加入到total集合中
                self.sortindex[flag].append([k,1]) # 新增一个排序项
                self.index[flag] = self.index[flag] + 1 # 最长下标自增
            else:
                # 加一之后更新项集所处的位置
                self.sortindex[flag][id][1] = self.insert[k] + self.delete[k]
                if id == 0 or self.sortindex[flag][id - 1][1] >= self.sortindex[flag][id][1]:
                    continue
                else:
                    for w in range(id, 0, -1):
                        if self.sortindex[flag][w][1] > self.sortindex[flag][w-1][1]:
                            self.total[self.sortindex[flag][w][0]] = w - 1
                            self.total[self.sortindex[flag][w-1][0]] = w
                            count, k = self.sortindex[flag][w][0], self.sortindex[flag][w][1]
                            self.sortindex[flag][w][0], self.sortindex[flag][w][1] = self.sortindex[flag][w-1][0], self.sortindex[flag][w-1][1]
                            self.sortindex[flag][w-1][0], self.sortindex[flag][w-1][1] = count, k
                        else:
                            break


    def InsertItem(self, item):
        # item = [[1,2],[2,4,5],[1,2],[2,4,5]]
        for i in item:
            # 对于item中的1项集，更新
            iter = itertools.combinations(i, 1)
            self.InsertOne(iter, 0)
            # 对于item中的2项集，更新
            if len(i) > 1:
                iter = itertools.combinations(i, 2)
                self.InsertOne(iter, 1)
            if len(i) > 2:
                for j in range(3, len(i) + 1):
                    iter = itertools.combinations(i, j)
                    self.InsertOne(iter, 2)

    def DeleteOne(self, iter, flag):

        for k in iter:
            # 插入项集
            self.insert[k] = self.insert.get(k) - 1 # 当前窗口频繁度减1
            self.delete[k] = self.gamma[flag] * (self.delete.setdefault(k, 0) + 1) # 折旧删除窗口并且频繁度加1
            id = self.total.get(k) # 找到项集的下标，往后更新
            self.sortindex[flag][id][1] = self.insert[k] + self.delete[k]  # 求得当前频繁度和折旧后窗口的加和
            for w in range(id, len(self.sortindex[flag])-1):
                if self.sortindex[flag][w][1] < self.sortindex[flag][w + 1][1]:
                    self.total[self.sortindex[flag][w][0]] = w + 1
                    self.total[self.sortindex[flag][w + 1][0]] = w
                    count, k = self.sortindex[flag][w][0], self.sortindex[flag][w][1]
                    self.sortindex[flag][w][0], self.sortindex[flag][w][1] = self.sortindex[flag][w + 1][0], \
                                                                             self.sortindex[flag][w + 1][1]
                    self.sortindex[flag][w + 1][0], self.sortindex[flag][w + 1][1] = count, k
                else:
                    break

    def DeleteItem(self, item):
        # item = [[1,2],[2,4,5],[1,2],[2,4,5]]
        for i in item:
            # 对于item中的1项集，更新
            iter = itertools.combinations(i, 1)
            self.DeleteOne(iter, 0)
            # 对于item中的2项集，更新
            if len(i) > 1:
                iter = itertools.combinations(i, 2)
                self.DeleteOne(iter, 1)
            if len(i) > 2:
                for j in range(3, len(i) + 1):
                    iter = itertools.combinations(i, j)
                    self.DeleteOne(iter, 2)

    '''
        返回一个numpy数组，维度为sum(k)*c_len，k是个数数组，有三维，分别代表每类项集的个数，c_len取61
        '''

    def FinfTop(self, k, c_len):
        res = numpy.zeros((sum(k), c_len))
        i = 0
        for j in range(len(k)):
            w = min(self.index[j], k[j])
            for x in range(w):
                for y in self.sortindex[j][x][0]:
                    res[i][y] = self.sortindex[j][x][1]
                i = i + 1
        return res

    def FindRate(self, item):
        id = len(item) - 1
        if id>2:
            id=2
        a = self.total.get(tuple(item))
        if a == None:
            return 0
        return self.sortindex[id][a][1]


