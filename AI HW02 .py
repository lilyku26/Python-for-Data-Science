
# coding: utf-8
"""
Created on Sun Apr 14 00:16:01 2019

@author: LitingKu
"""
# In[8]:


import numpy as np
import string
import random
from memory_profiler import profile



i = eval(input("請輸入行數:"))
j = eval(input("請輸入列數:"))
p = eval(input("請輸入蜂蜜數量:"))
a = eval(input("希望蜜蜂初始點在第幾行(<總行數):"))
b= eval(input("希望蜜蜂初始點在第幾列(<總列數):"))
ll = np.random.choice(['x','.'],(i,j))
print("此程式只能生成一組捷徑")



ll = np.random.choice(['x','.'],(i,j))
#蜜蜂位置
ll[a-1,b-1]='b' 
#隨機取蜂蜜的所在位置
list1= [ir for ir in range(i)]
list2= [ir for ir in range(j)]
k=ll[ll=='h']
while len(k)!=p:
    g=random.sample(list1,p)
    f=random.sample(list2,p)
    for ii in range(0,p):
        if ll[g[ii],f[ii]]!='b':
            ll[g[ii],f[ii]]='h'
    k = ll[ll=='h']

m= ll[ll=='1']
while len(m)!=2:
    g=random.sample(list1,2)
    f=random.sample(list2,2)
    for ii in range(0,2):
        if ll[g[ii],f[ii]]not in('b','h'):
            ll[g[ii],f[ii]]='1'
    m = ll[ll=='1']
ll


# In[13]:


from enum import Enum
import copy
import time

# Define state ------------------------------------------------------------------
class EndType(Enum):
    END = 0
    FINISH = 1

class Walked(Enum):
    UNWALKED = 0
    WALKED = 1

class PointType(Enum):
    ROAD = 0
    WALL = 1
    TRANSFER = 2
    BEE = 3
class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    RIGHT_UP = 4
    RIGHT_DOWN = 5
    LEFT_UP = 6
    LEFT_DOWN = 7
    def __str__(self):
        return ['↑', '↓',' ← ', ' → ', '↗', '↘', '↖', '↙'][self.value]
    
class Location(object):
    def __init__(self, row, col):
        self.row=row
        self.col=col
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    def __ne__(self, other):
        return self.row != other.row or self.col != other.col
    def getDirectionLocation(self, direct):
        loc = Location(self.row, self.col)
        if direct==Direction.UP:
            loc.getUp()
        elif direct==Direction.DOWN:
            loc.getDown()
        elif direct==Direction.LEFT:
            loc.getLeft()
        elif direct==Direction.RIGHT:
            loc.getRight()
        elif direct==Direction.RIGHT_UP: 
            loc.getRightUp()
        elif direct==Direction.RIGHT_DOWN: 
            loc.getRightDown()
        elif direct==Direction.LEFT_UP: 
            loc.getLeftUp()
        elif direct==Direction.LEFT_DOWN: 
            loc.getLeftDown()
        return loc
    def getUp(self):
        self.row-=1
    def getDown(self):
        self.row+=1
    def getLeft(self):
        self.col-=1
    def getRight(self):
        self.col+=1
    def getRightUp(self):
        self.col+=1
        self.row-=1
    def getRightDown(self):
        self.col+=1
        self.row+=1
    def getLeftUp(self):
        self.col-=1
        self.row-=1
    def getLeftDown(self):
        self.col-=1
        self.row+=1
#----------------------------------------------------------------------------------------------
        
# Heuristic Function --------------------------------------------------------------------------
    def getDistance(self, other):
#  "Manhattan distance"
#function heuristic(node) =
#   dx = abs(node.x - goal.x)
#   dy = abs(node.y - goal.y)
#   return D * (dx + dy)
# let D=1
#        return int(np.abs(self.row - other.row) + np.abs(self.col - other.col))

#  "Euclidean distance"    
#        return int(np.sqrt((self.row - other.row)**2 + (self.col - other.col)**2)) 

#  "Diagonal distance"
#function heuristic(node) =
#   dx = abs(node.x - goal.x)
#   dy = abs(node.y - goal.y)
#   return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
# if D=1 D2=1  → Chebyshev distance
# if D=1 D2 = sqrt(2) → octile distance
         return int(np.abs(self.row - other.row)+ np.abs(self.col - other.col) - min(np.abs(self.row - other.row),np.abs(self.col - other.col)))
#------------------------------------------------------------------------------------------------

    def __str__(self):
        #return f"({self.row}, {self.col})"
        return "("+str(self.row)+", "+str(self.col)+")"
    
# 設定程式讀取一開始創造出的蜂巢矩陣的參數 -----------------------------------------------------------------
class Point(object):
    def __init__(self, row, col, ptype):
        self.loc=Location(row, col)
        self.type=ptype
    def __str__(self):
        return " "
        
class Road(Point):
    def __init__(self, row, col):
        super().__init__(row, col, PointType.ROAD)
    def __str__(self):
        return "."
        
class Honey(object):
    def __init__(self, row, col):
        self.loc=Location(row, col)
    def __str__(self):
        return "H:"+str(self.loc)

class BEE(Point):
    def __init__(self, row, col):
        super().__init__(row, col, PointType.BEE)
    def __str__(self):
        return "b"

    
class Wall(Point):
    def __init__(self, row, col):
        super().__init__(row, col, PointType.WALL)
    def __str__(self):
        return "x"
# 捷徑
class Transfer(Point):
    def __init__(self, row, col, tid):
        super().__init__(row, col, PointType.TRANSFER)
        self.trans=Location(-1, -1)#Transfer(-1, -1, int(tid))
        self.id=int(tid)
    def setTarget(self, row, col):
        self.trans.row=row
        self.trans.col=col
    def getTargetLocation(self):
        return self.trans
    def __str__(self):
        return str(self.id)

#建構蜂巢 Hive -----------------------------------------------------------------
class Hive(object):
    def __init__(self, Map, size):
        #為一開始生成的矩陣加上牆壁
        self.sizeRow=np.size(ll,0)+2 
        self.sizeCol=np.size(ll,1)+2
        self.bee=Bee(0, 0)
        self.honey=[]
        transfer=[]
        self.map=[]
        for i in range(self.sizeRow):
            self.map.append([])
            for j in range(self.sizeCol):
                if i== 0 or i==self.sizeRow-1 or j== 0 or j==self.sizeCol-1:
                    self.map[i].append(Wall(i, j))
                else:
                    if Map[i-1][j-1]=='x':
                        self.map[i].append(Wall(i, j))
                    elif Map[i-1][j-1]=='.':
                        self.map[i].append(Road(i, j))
                    elif Map[i-1][j-1]=='b':
                        self.map[i].append(BEE(i, j))
                        self.bee.setLocation(i, j)
                    elif Map[i-1][j-1]=='h':
                        self.map[i].append(Road(i, j))
                        self.honey.append(Honey(i, j))
                    else:
                        self.map[i].append(Transfer(i, j, Map[i-1][j-1]))
                        find=False
                        for index in range(len(transfer)):
                            if transfer[index]['id']==Map[i-1][j-1]:
                                self.map[i][j].setTarget(transfer[index]['row'], transfer[index]['col'])
                                self.map[transfer[index]['row']][transfer[index]['col']].setTarget(i, j)
                                del transfer[index]
                                find=True
                        if not find:
                            transfer.append({'id':Map[i-1][j-1], 'row':i, 'col':j})

    def getMap(self, loc):
        return self.map[loc.row][loc.col]
    def getLocation(self, loc):
        if self.map[loc.row][loc.col].type==PointType.TRANSFER:
            return self.map[loc.row][loc.col].getTargetLocation()
        return loc
#--------------------------------------------------------------------------------------------------------------
# 蜂蜜的位置
    def isHoney(self, loc):
        for honey in self.honey:
            if honey.loc==loc:
                return True
        return False
    
#把走過的蜂蜜給去除，避免重複
    def collectHoney(self, loc):
        for honey in self.honey:
            if honey.loc==loc:
                self.honey.remove(honey)
    def __str__(self):
        string=""
        for x in range(self.sizeRow):
            for y in range(self.sizeCol):
                if self.isHoney(Location(x, y)):
                    string+="h "
                else:
                    string+=str(self.map[x][y])+" "
            string+="\n"
        return string
    def printHoney(self, honeyList):
        for honey in self.honey:
            honey.display()
            
 #小蜜蜂與蜂蜜的距離
    def __getMinHoney(self, start, honeyList):
        dis=[start.getDistance(honey.loc) for honey in honeyList]
        ind=np.argmin(np.asarray(dis))
        return ind, dis[ind]
    def heuristic(self, begin):
        cost=0
        honeyList=copy.deepcopy(self.honey)
        start=copy.copy(begin)
        while len(honeyList)>0:
            ind, dis=self.__getMinHoney(start, honeyList)
            cost+=dis
            start=honeyList[ind].loc
            honeyList.pop(ind)
        return int(cost)

#蜜蜂移動的方向與位置
class Bee(object):
    def __init__(self, row, col):
        self.loc=Location(row, col)
    def setLocation(self, row, col):
        self.loc=Location(row, col)
    def move(self, direct):
        if direct==Direction.UP:
            self.loc.getUp()
        elif direct==Direction.DOWN:
            self.loc.getDown()
        elif direct==Direction.RIGHT_UP: 
            self.loc.getRightUp()
        elif direct==Direction.RIGHT_DOWN: 
            self.loc.getRightDown()
        elif direct==Direction.LEFT_UP: 
            self.loc.getLeftUp()
        elif direct==Direction.LEFT_DOWN: 
            self.loc.getLeftDown()
    def __str__(self):
        print('BEE:', end='')
        self.loc.display()

def initWalked(hive, start):
    walked = [[Walked.UNWALKED for j in range(hive.sizeCol)] for i in range(hive.sizeRow)]
    walked[start.row][start.col]=Walked.WALKED
    return walked
    
def getMinCost(cost):
    min_index=np.argmin(np.asarray(cost))
    return min_index, cost[min_index]
#-------------------------------------------------------------------------------------------------
# Deep limit search ------------------------------------------------------------------------------
def DLS(hive, walked, start, limit):
    if limit <= 0:
        return [EndType.END], int(infinite)
    start=hive.getLocation(start)
    if hive.getMap(start).type==PointType.WALL or walked[start.row][start.col]!=Walked.UNWALKED:
        return [EndType.END], int(infinite)
    walked[start.row][start.col]=Walked.WALKED
    if (hive.isHoney(start)):
        hive.collectHoney(start)
        if len(hive.honey)>0:
            walked = initWalked(hive, start)
        else:
            return [EndType.FINISH], 0
    new_walked = copy.deepcopy(walked)
    costList = []
    pathList = []
    for direct in list(Direction):
        new_hive=copy.deepcopy(hive)
        path, cost=DLS(new_hive, new_walked, start.getDirectionLocation(direct), limit-1)
        costList.append(cost)
        pathList.append(path)
    dir_index, step=getMinCost(costList)
    route = pathList[dir_index]
    route.append(Direction(dir_index))
    return route, step+1
#-------------------------------------------------------------------------------------------------

# IDS algorithm-----------------------------------------------------------------------------------
@profile(precision=6)
def IDS(hive, maxDepth):
    start=hive.bee.loc
    for limit in range(maxDepth):  #iterating depth
        new_hive=copy.deepcopy(hive)
        costList = []
        routeList = []
        for direct in list(Direction):
            route, cost=DLS(new_hive, initWalked(hive, hive.bee.loc), start.getDirectionLocation(direct), limit)
            costList.append(cost)
            routeList.append(route)
        dir_index, step=getMinCost(costList)
        route = routeList[dir_index]
        route.append(Direction(dir_index))
        if route[0]==EndType.FINISH:
            return route, step+1
    print('No FOUND')
    return route, step+1

def Search_IDS(hive, maxDepth):
    print('IDS')
    startTime=time.time()
    route, cost = IDS(hive, maxDepth)
    endTime=time.time()
    printRoute(route)
    print('An optimal solution has ', cost,' steps.')
    print('Total run time=', endTime-startTime, 'seconds.')
#----------------------------------------------------------------------------------------------------------
    

# A* algorithm ---------------------------------------------------------------------------------------------
def Astar(hive, walked, start, limit, gn):
    start=hive.getLocation(start)
    if hive.getMap(start).type==PointType.WALL or walked[start.row][start.col]!=Walked.UNWALKED:
        return [EndType.END], int(infinite)
    hn=hive.heuristic(start)
    if limit < hn+gn:
        return [EndType.END], hn+gn
    walked[start.row][start.col]=Walked.WALKED
    if (hive.isHoney(start)):
        hive.collectHoney(start)
        if len(hive.honey)>0:
            walked = initWalked(hive, start)
        else:
            return [EndType.FINISH], gn+1
    new_walked = copy.deepcopy(walked)
    costList = []
    pathList = []
    for direct in list(Direction):
        new_hive=copy.deepcopy(hive)
        path, cost=Astar(new_hive, new_walked, start.getDirectionLocation(direct), limit, gn+1)
        costList.append(cost)
        pathList.append(path)
    dir_index, next_limit=getMinCost(costList)
    route = pathList[dir_index]
    route.append(Direction(dir_index))
    return route, next_limit
#-------------------------------------------------------------------------------------------------------------------

# IDA* alogorithm --------------------------------------------------------------------------------------------------
@profile(precision=6)
def IDAstar(hive, maxDepth):
    start=hive.bee.loc
    limit=hive.heuristic(start)
    while limit<=maxDepth:
        new_hive=copy.deepcopy(hive)
        costList = []
        routeList = []
        for direct in list(Direction):
            route, cost=Astar(new_hive, initWalked(hive, hive.bee.loc), start.getDirectionLocation(direct), limit, 0)
            costList.append(cost)
            routeList.append(route)
        dir_index, next_limit=getMinCost(costList)
        route = routeList[dir_index]
        route.append(Direction(dir_index))
        limit=next_limit
        if route[0]==EndType.FINISH:
            return route, next_limit
    print('No FOUND')
    return route, next_limit
    
def Search_IDAstar(hive, maxDepth):
    print('IDAstar')
    startTime=time.time()
    route, cost = IDAstar(hive, maxDepth)
    endTime=time.time()
    printRoute(route)
    print('An optimal solution has ', cost,' steps.')
    print('Total run time=', endTime-startTime, 'seconds.')
def printRoute(route):
    for step in range(len(route)-1, 0, -1):
        print(route[step], end='')
    print()

# - --------------------------------------------------------- -------------------------------------    
global direction
global infinite
infinite=1e10
maxDepth=100
hive=Hive(ll, np.shape(ll))
print(hive)
print('Heruisrtic function : Diagonal distance')
Search_IDAstar(hive, maxDepth)
Search_IDS(hive, maxDepth)


