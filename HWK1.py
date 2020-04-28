import json
import copy
import queue
from queue import PriorityQueue
from collections import defaultdict
from enum import Enum

GOALSTATE = [[]]  #[0,0,0,
                # 0,0,0,
                # 0,0,0]
BOUND = 1  # 31 moves to solve any 8 puzzle and 80 moves to solve any 15 puzzle

# This is what I use to replicate an ENUM for now
class State():
    FAIL = 0
    NIL = 1


class GraphNode():
    def __init__(self):
        self.currentState = [[]]
        self.heuristic = 0
        self.next1 = None
        self.next2 = None
        self.next3 = None
        self.next4 = None
        self.name = 0
        # max connections is 4 connections

    def printGrid(self, drawArrow):
        # This one is different since I don't have a size variable
        if drawArrow == True:
            print("\t/\\")
            print("\t|")
        for i in range(len(self.currentState)):
            print(self.currentState[i])

    def __lt__(self, other):
        return self.name < other.name


class Puzzle:
    def __init__(self):
        self.movesToSolve = 0
        self.size = 0
        self.currentState = []   #[0,0,0,
                                 # 0,0,0,
                                 # 0,0,0]
        self.rules = Rules()
        self.returnTypes = State()

    def readJsonFile(self, fileName):
        # https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
        with open(fileName) as json_file:
            data = json.load(json_file)
        return data

    # This is the function that will make sure the json file is a valid one
    def setUp(self, jsonData):
        # First error checking, check if ALL the fields exists in the first place, then check value, if it all passed, store it
        if "n" not in jsonData:
            print("Json file does not have a field for n")
            return False
        if "start" not in jsonData:
            print("Json file does not have a field for start")
            return False
        if "goal" not in jsonData:
            print("Json file does not have a field for goal")
            return False
        # Second error checking, the actual values inside of the json file
        # We know the fields exist right now
        if jsonData["n"] < 1: # If this is true, there is an error with the value
            print("Json file does not have a valid number for n")
            return False
        else:
            self.size = jsonData["n"];

        # -------------------------------------------------------------------------------
        # at this point we know that all fields have a value in them
        global GOALSTATE
        #global BOUND
        #if self.size == 3:
        #    BOUND = 31
        #elif self.size == 4:
        #    BOUND = 80
        gridList = []
        gridInt = []
        current = jsonData["start"]
        goal = jsonData["goal"]
        if validateGrid1(current, self.size): # if this is true, the grid is valid
            self.currentState = jsonData["start"]
            #for i in range(self.size):
            #    list1 = current[i]
            #    gridList += list1
            # Let's convert the list into ints to be able to comparegoal = jsonData["goal"]
            #for i in range(self.size * self.size):
            #    self.currentState.append(int(gridList[i]))
        else:
            print("Json file does not have a valid grid for start")
            return False
        if validateGrid1(goal, self.size):
            GOALSTATE = jsonData["goal"]
            #gridList.clear()
            #for i in range(self.size):
            #    list1 = goal[i]
            #    gridList += list1
            # Let's convert the list into ints to be able to compare
            #for i in range(self.size * self.size):
            #    GOALSTATE.append(int(gridList[i]))
        else:
            print("Json file does not have a valid grid for goal")
            return False
        # if we are at this point, we have past all the checks and the json file is valid
        return True

    def displayGrid(self):
        print("Current grid:\t\t\tGoal State:")
        for i in range(self.size):
            print(self.currentState[i], end='')
            print("\t\t\t\t", GOALSTATE[i])
        #for i in range(self.size*self.size):
        #    if i % self.size is 0:
        #        print("[", self.currentState[i], " ", end='')
        #    elif i % self.size is self.size - 1:
        #        print(self.currentState[i], "]")
        #    else:
        #        print(self.currentState[i], " ", end='')


class Rules:
    def __init__(self):
        self.up = "up"
        self.down = "down"
        self.right = "right"
        self.left = "left"

    def applicableRules(self, puzzleGrid):
        #Find the index of 0, that will tell us where our empty spot is
        #We know the grid is valid so we know there is a 0, we can just check for it instead of checking if it exists
        indexI = getWhiteSpaceIndexI(puzzleGrid)
        indexJ = getWhiteSpaceIndexJ(puzzleGrid)
        #[4,1,2]
        #[3,0,5]
        #[6,7,8]
        rules = []
        size = len(puzzleGrid)

        if indexI - 1 >= 0:
            rules.append(self.up)
        if indexJ + 1 < size:
            rules.append(self.right)
        if indexI + 1 < size:
            rules.append(self.down)
        if indexJ - 1 >= 0:
            rules.append(self.left)
        return rules

    def applyRule(self, puzzleGrid, rule):
        if rule is self.up:
            return self.applyUp(puzzleGrid)
        if rule is self.right:
            return self.applyRight(puzzleGrid)
        if rule is self.down:
            return self.applyDown(puzzleGrid)
        if rule is self.left:
            return self.applyLeft(puzzleGrid)

    # When you apply a rule, make sure the grid is being coppied and you return the new grid
    def applyUp(self, puzzleGrid):
        size = len(puzzleGrid)
        indexI = getWhiteSpaceIndexI(puzzleGrid)
        indexJ = getWhiteSpaceIndexJ(puzzleGrid)
        newIndex = indexI - 1
        value = puzzleGrid[newIndex][indexJ]
        copiedGrid = copy.deepcopy(puzzleGrid)
        # copiedGrid = puzzleGrid.copy()
        copiedGrid[newIndex][indexJ] = 0
        copiedGrid[indexI][indexJ] = value
        return copiedGrid

    def applyDown(self, puzzleGrid):
        size = len(puzzleGrid)
        indexI = getWhiteSpaceIndexI(puzzleGrid)
        indexJ = getWhiteSpaceIndexJ(puzzleGrid)
        newIndex = indexI + 1
        value = puzzleGrid[newIndex][indexJ]
        copiedGrid = copy.deepcopy(puzzleGrid)
        # copiedGrid = puzzleGrid.copy()
        copiedGrid[newIndex][indexJ] = 0
        copiedGrid[indexI][indexJ] = value
        return copiedGrid

    def applyRight(self, puzzleGrid):
        size = len(puzzleGrid)
        indexI = getWhiteSpaceIndexI(puzzleGrid)
        indexJ = getWhiteSpaceIndexJ(puzzleGrid)
        newIndex = indexJ + 1
        value = puzzleGrid[indexI][newIndex]
        copiedGrid = copy.deepcopy(puzzleGrid)
        # copiedGrid = puzzleGrid.copy()
        copiedGrid[indexI][newIndex] = 0
        copiedGrid[indexI][indexJ] = value
        return copiedGrid

    def applyLeft(self, puzzleGrid):
        size = len(puzzleGrid)
        indexI = getWhiteSpaceIndexI(puzzleGrid)
        indexJ = getWhiteSpaceIndexJ(puzzleGrid)
        newIndex = indexJ - 1
        value = puzzleGrid[indexI][newIndex]
        copiedGrid = copy.deepcopy(puzzleGrid)
        # copiedGrid = puzzleGrid.copy()
        copiedGrid[indexI][newIndex] = 0
        copiedGrid[indexI][indexJ] = value
        return copiedGrid


# No longer in use since we are no longer using a simple list
def validateGrid(grid, size):
    numberArray = []    # will use this to store the numbers inside the grid
    gridList = []
    for i in range(size):
        list1 = grid[i]
        gridList += list1
    # list1 = grid[0] #these are 3x3 grids
    # list2 = grid[1]
    # list3 = grid[2]
    # gridList = list1 + list2 + list3
    gridInt = []
    # Let's convert the list into ints to be able to compare
    for i in range(size*size):
        gridInt.append(int(gridList[i]))

    for i in range(size*size):
        if gridInt[i] >= 0 and gridInt[i] <= ((size*size) - 1):   # this means it's a valid number in the grid
            if gridInt[i] not in numberArray:
                numberArray.append(gridInt[i]);
            else:
                return False
        else:
            return False
    # if we're here, there was no duplicates and all the numbers were fine
    return True

# Use this function since we are using a vector of vectors
def validateGrid1(grid,size):
    # The grid is recieved as a vector of vectors
    numberArray = []    # we will use this array to check if there is a duplicate number in the grid or not
    for i in range(size):
        for j in range(size):
            if grid[i][j] >= 0 and grid[i][j] <= ((size * size) - 1):
                if grid[i][j] not in numberArray:
                    numberArray.append(grid[i][j]);
                else:
                    return False
            else:
                return False
    # if we're here, there was no duplicates and all the numbers were fine
    return True


def getWhiteSpaceIndexI(grid):
    size = len(grid)    # we know that the grid is always X by X and not X by Y so we can assume that if length is 4, width is 4 as well
    for i in range(size):
        for j in range(size):
            if grid[i][j] == 0:
                index = i
    return index


def getWhiteSpaceIndexJ(grid):
    size = len(grid)    # we know that the grid is always X by X and not X by Y,we can assume that if length is 4, width is 4
    for i in range(size):
        for j in range(size):
            if grid[i][j] == 0:
                index = j
    return index


def getIndexI(grid, number):
    size = len(grid)  # we know that the grid is always X by X and not X by Y so we can assume that if length is 4, width is 4 as well
    for i in range(size):
        for j in range(size):
            if grid[i][j] == number:
                index = i
    return index


def getIndexJ(grid, number):
    size = len(grid)  # we know that the grid is always X by X and not X by Y so we can assume that if length is 4, width is 4 as well
    for i in range(size):
        for j in range(size):
            if grid[i][j] == number:
                index = j
    return index


def printWinningPath(rulesApplied, stateExamined):
    step = 1
    print("")  # I'm going to use this for a new line
    print("Solution: ")
    for i in range(len(rulesApplied) - 1, -1, -1):
        print(step, ".", rulesApplied[i])
        step += 1
    print("Examined ", stateExamined, " total amount of states!")


def addIntoDictionary(stateToAdd, dict, currentState):
    match = False
    for k, v in dict.items():
        if k == currentState.currentState and v == stateToAdd.currentState:
            match = True
    if match == False:
        dict.update({copy.deepcopy(currentState): copy.deepcopy(stateToAdd)})


def backtrack1(datalist, ruleObj, stateObj, rulesApplied, stateExamined):
    number = stateExamined[0]   # I'm using an array to simulate a pass by reference with my number variable
    number += 1
    stateExamined[0] = number
    # Grab the newest data, which will always be at the 0th index
    data = datalist[0]  # This grabs a grid
    # Check data against the entire list to make sure it's not repeating an earlier data
    for i in range(len(datalist))[1:]:  # This will start at i = 1
        if data == datalist[i]:
            return stateObj.FAIL
    # Check if we have a match with the goal
    if data == GOALSTATE:
        return stateObj.NIL
    # if the length of the datalist is greater than the depth bound limit, return fail
    if len(datalist) > BOUND:
        return stateObj.FAIL
    # Grab the rules that we can apply to the current grid
    ruleList = ruleObj.applicableRules(data)
    # Loop here
    loopOver = False
    while loopOver is False:
        # if we have no rules from above, return fail
        if not ruleList:    # this checks if the list is null/empty
            return stateObj.FAIL
        # else grab the first rule from the list of rules
        R = ruleList[0]     # 0th index is always going to be the first
        ruleList.pop(0)
        # Create the new grid by applying the rules
        newGrid = ruleObj.applyRule(data,R)
        rulesApplied.insert(0, R)
        # Add the new grid into the new list of grids
        newDatalist = datalist.copy()
        newDatalist.insert(0, newGrid)
        # call backtrack1 and store the return value
        returnValue = backtrack1(newDatalist, ruleObj, stateObj, rulesApplied, stateExamined)
        # if variable == fail, go loop
        if returnValue is not stateObj.FAIL:
            loopOver = True
        # else it was a fail and we remove the rule we applied
        else:
            rulesApplied.pop(0)

    return rulesApplied    # We want to return a list of all the rules that were applied
    # technically it's always updated and we don't need to return but oh well
    # End loop here


def graphSearch(currentState, rules, state):
    # When you create the successors, a shortcut is to add all of those nodes to open
    # It will take longer to solve but it will still solve the solution
    # Create first node
    nodeS = GraphNode()
    nodeS.currentState = copy.deepcopy(currentState)     # we use a deep copy to not affect the current state
    name = 0
    nodeS.name = name
    name += 1
    nodeS.next1 = None
    # Create open queue and fill it
    open = queue.Queue()
    open.put(nodeS) # we place the first state in the queue
    # Create closed queue
    closed = queue.Queue()
    # Create variables for the loop
    path = []
    loopOver = False
    stateToAdd = GraphNode()
    currentState = GraphNode()
    m = []  # this will be the list of all successors of the current state
    statesGenerated = 0
    statesExplored = 0
    while loopOver == False:
        if open.empty():    # this means that there are no more open states left and we failed the find a solution
            print("Failed :(")
            return state.FAIL
        # Retrieve a node and expand it
        n = open.get(0) # retrieve a state from open and get rid of it from open
        statesExplored += 1
        closed.put(n)   # add the state we retrieved to closed
        path.append(n.currentState)
        if n.currentState == GOALSTATE:  # we found a solution if this is true
            # Display length of path and then return
            pathCost = 0
            while n.next1 != None:
                pathCost += 1
                if pathCost == 1:
                    n.printGrid(False)  # The false is for an arrow, I know this is misleading, sorry
                else:
                    n.printGrid(True)
                print()
                n = n.next1
            nodeS.printGrid(True)
            print()
            print("Total cost of path: ", pathCost)
            print("States generated: ", statesGenerated)
            print("States explored: ", statesExplored)
            return state.NIL
        ruleList = rules.applicableRules(n.currentState)
        # Creating all successors for expansion
        while len(ruleList) > 0:
            r = ruleList[0]
            ruleList.pop(0)
            successor = rules.applyRule(n.currentState, r)
            m.append(successor)
            statesGenerated += 1
        # m is a set of all successors of n currently
        while len(m) > 0:
            # if the successor already exists in closed, don't add it
            if m[0] not in path:    # using path because Queue is not iterable
                # The successor is not in closed so we do want to add it to open
                currentState.currentState = copy.deepcopy(m[0])
                # make it point back to n
                currentState.next1 = n
                currentState.name = name
                name += 1
                stateToAdd = copy.deepcopy(currentState)
                open.put(stateToAdd)
                m.pop(0)
            else:
                m.pop(0)


def wrongTilesAmount(currentState):
    wrongTile = 0
    for i in range(len(currentState)):
        for j in range(len(currentState[i])):
            if currentState[i][j] != GOALSTATE[i][j]:
                wrongTile += 1
    return wrongTile


def manhattanDistance(currentState):
    rowSize = len(currentState)
    columnSize = len(currentState[0])
    manhattanNumber = 0
    for i in range(rowSize):
        for j in range(columnSize):
            if currentState[i][j] != GOALSTATE[i][j]:
                # calculate the manhattan distance of this tile
                if currentState[i][j] != 0:
                    currentIndexJ = j
                    currentIndexI = i
                    indexI = getIndexI(currentState, GOALSTATE[i][j])
                    indexJ = getIndexJ(currentState, GOALSTATE[i][j])
                    distanceI = abs(indexI - currentIndexI)
                    distanceJ = abs(indexJ - currentIndexJ)
                    manhattanNumber += distanceI + distanceJ
    return manhattanNumber


def checkExistance(open, closed, currentGrid):
    openListObj = []
    openListState = []
    closedListObj = []
    closedListState = []
    index = 0
    found = False
    while not open.empty():
        openListObj.append(open.get())
        a = openListObj[index]  # this is the obj itself
        openListState.append(a[1].currentState)
        index += 1
    # check if grid is in open
    if currentGrid in openListState:
        found = True
    index = 0
    while not closed.empty():
        closedListObj.append(closed.get())
        a = closedListObj[index].currentState
        closedListState.append(a)
    # check if grid is in closed
    index = 0
    if currentGrid in closedListState:
        found = True

    # restore queues
    while len(openListObj) != index:
        open.put(openListObj[index])
        index += 1
    index = 0
    while len(closedListObj) != index:
        closed.put(closedListObj[index])
        index += 1

    return found


def compareAgainstExistingNode(open, closed, n, currentNode):
    openListObj = []
    openListState = []
    closedListObj = []
    closedListState = []
    index = 0
    while not open.empty():
        openListObj.append(open.get())
        a = openListObj[index]
        openListState.append(a[1].currentState)
        index += 1
    if currentNode.currentState in openListState:
        indexOfMatch = openListState.index(currentNode.currentState)
        oldNode = openListObj[indexOfMatch]
        if oldNode[1].heuristic > currentNode.heuristic:
            oldNode[1].next1 = n
            openListObj[indexOfMatch] = oldNode
    index = 0
    while not closed.empty():
        closedListObj.append(closed.get())
        a = closedListObj[index].currentState
        closedListState.append(a)
        index += 1
    if currentNode.currentState in closedListState:
        indexOfMatch = closedListState.index(currentNode.currentState)
        oldNode = closedListObj[indexOfMatch]
        if oldNode.heuristic > currentNode.heuristic:
            oldNode.next1 = n
            closedListObj[indexOfMatch] = oldNode
    index = 0
    # restore everything now
    while len(openListObj) != index:
        open.put(openListObj[index])
        index += 1
    index = 0
    while len(closedListObj) != index:
        closed.put(closedListObj[index])
        index += 1
    return


def displayPath(closed):
    # Display length of path and then return
    pathCost = 0
    moves = 0
    closedList = []
    while not closed.empty():
        closedList.append(closed.get())
    # n currently is the goal state
    size = len(closedList)
    n = closedList[size - 1]    # this is the goal state
    while n.next1 != None:
        pathCost += n.heuristic
        moves += 1
        if pathCost == n.heuristic:  # This will only be false for the very first drawing
            n.printGrid(False)  # The false is for an arrow, I know this is misleading, sorry
        else:
            n.printGrid(True)
        print()
        # matchingIndex = closedList.index(n.next1)
        # n = closedList[matchingIndex]
        n = n.next1
    pathCost += n.heuristic  # this is needed since the first heustic of the loop was 0 (goal)
    n.printGrid(True)
    print()
    print("Total heuristic cost of path: ", pathCost)
    print("Total moves: ", moves)
    return

# when you have a successor, if it's in open or closed, check if it's a better node, if it is, update the already
# existing node to have the parent pointer point to the current n
# update closed with compareAgainstExistingNode to repoint pointers, pop the queue and follow
# pointers to find optimal path
def aStarAlgH1(currentState, rules, state):
    # very similar to graph search except we will be using a priority queue and H(n)
    # H(n) = H1(n) + H2(n) which will be the amount of wrong tiles and the manhattan distance
    name = 0    # will increment, this will be used for my priority queue to never have a duplicate
    nodeS = GraphNode()
    nodeS.currentState = copy.deepcopy(currentState)  # we use a deep copy to not affect the current state
    nodeS.name = name
    h1 = wrongTilesAmount(currentState)
    nodeS.heuristic = h1
    nodeS.next1 = None
    name += 1

    # Create open priority queue and fill it
    open = PriorityQueue()
    open.put((nodeS.heuristic, nodeS))  # we place the first state in the queue with h(n) being the priority
    # Create closed queue
    closed = queue.Queue()

    # Create variables for the loop
    path = []
    possibleNodes = []
    loopOver = False
    heuristicDuplicates = True
    stateToAdd = GraphNode()
    currentState = GraphNode()
    m = []  # this will be the list of all successors of the current state
    statesGenerated = 0
    statesExplored = 0

    while loopOver == False:
        if open.empty():  # this means that there are no more open states left and we failed the find a solution
            print("Failed :(")
            return state.FAIL
        # Retrieve a node and expand it
        a = open.get()
        # since I copied graph search and everything uses n, this was easier to to instead of replacing every n with n[1]
        n = a[1]
        statesExplored += 1
        closed.put(n)  # add the state we retrieved to closed
        path.append(n.currentState)
        if n.currentState == GOALSTATE:  # we found a solution if this is true
            # Display length of path and then return
            pathCost = 0
            moves = 0
            while n.next1 != None:
                pathCost += n.heuristic
                moves += 1
                if pathCost == n.heuristic: # This will only be false for the very first drawing
                    n.printGrid(False)  # The false is for an arrow, I know this is misleading, sorry
                else:
                    n.printGrid(True)
                print()
                n = n.next1
            pathCost += nodeS.heuristic # this is needed since the first heustic of the loop was 0 (goal)
            nodeS.printGrid(True)
            print()
            print("Total heuristic cost of path: ", pathCost)
            print("Total moves: ", moves)
            print("States generated: ", statesGenerated)
            print("States explored: ", statesExplored)
            return state.NIL
        ruleList = rules.applicableRules(n.currentState)
        # Creating all successors for expansion
        while len(ruleList) > 0:
            r = ruleList[0]
            ruleList.pop(0)
            successor = rules.applyRule(n.currentState, r)
            m.append(successor)
            statesGenerated += 1
        # m is a set of all successors of n currently
        while len(m) > 0:
            # if the successor already exists in closed, check if the current one has a smaller h(n)
            if m[0] not in path:  # using path because Queue is not iterable
                # The successor is not in closed so we do want to add it to open
                currentState.currentState = copy.deepcopy(m[0])
                # make it point back to n
                currentState.next1 = n
                currentState.name = name
                name += 1
                h1 = wrongTilesAmount(currentState.currentState)
                #h2 = manhattanDistance(currentState.currentState)
                currentState.heuristic = h1
                stateToAdd = copy.deepcopy(currentState)
                open.put((stateToAdd.heuristic, stateToAdd))
                m.pop(0)
            else:
                # check against already existing node
                m.pop(0)


def aStarAlgH2(currentState, rules, state):
    # very similar to graph search except we will be using a priority queue and H(n)
    # H(n) = H1(n) + H2(n) which will be the amount of wrong tiles and the manhattan distance
    name = 0    # will increment, this will be used for my priority queue to never have a duplicate
    # h1 = 0
    h2 = 0
    nodeS = GraphNode()
    nodeS.currentState = copy.deepcopy(currentState)  # we use a deep copy to not affect the current state
    nodeS.name = name
    # h1 = wrongTilesAmount(currentState)
    h2 = manhattanDistance(currentState)
    nodeS.heuristic = h2# h1 + h2
    nodeS.next1 = None
    name += 1

    # Create open priority queue and fill it
    open = PriorityQueue()
    open.put((nodeS.heuristic, nodeS))  # we place the first state in the queue with h(n) being the priority
    # Create closed queue
    closed = queue.Queue()

    # Create variables for the loop
    path = []
    possibleNodes = []
    loopOver = False
    heuristicDuplicates = True
    stateToAdd = GraphNode()
    currentState = GraphNode()
    m = []  # this will be the list of all successors of the current state
    statesGenerated = 0
    statesExplored = 0

    while loopOver == False:
        if open.empty():  # this means that there are no more open states left and we failed the find a solution
            print("Failed :(")
            return state.FAIL
        # Retrieve a node and expand it
        a = open.get()
        # since I copied graph search and everything uses n, this was easier to to instead of replacing every n with n[1]
        n = a[1]
        statesExplored += 1
        closed.put(n)  # add the state we retrieved to closed
        path.append(n.currentState)
        if n.currentState == GOALSTATE:  # we found a solution if this is true
            # Display length of path and then return
            pathCost = 0
            moves = 0
            while n.next1 != None:
                pathCost += n.heuristic
                moves += 1
                if pathCost == n.heuristic: # This will only be false for the very first drawing
                    n.printGrid(False)  # The false is for an arrow, I know this is misleading, sorry
                else:
                    n.printGrid(True)
                print()
                n = n.next1
            pathCost += nodeS.heuristic # this is needed since the first heustic of the loop was 0 (goal)
            nodeS.printGrid(True)
            print()
            print("Total heuristic cost of path: ", pathCost)
            print("Total moves: ", moves)
            print("States generated: ", statesGenerated)
            print("States explored: ", statesExplored)
            return state.NIL
        ruleList = rules.applicableRules(n.currentState)
        # Creating all successors for expansion
        while len(ruleList) > 0:
            r = ruleList[0]
            ruleList.pop(0)
            successor = rules.applyRule(n.currentState, r)
            m.append(successor)
            statesGenerated += 1
        # m is a set of all successors of n currently
        while len(m) > 0:
            # if the successor already exists in closed, check if the current one has a smaller h(n)
            if m[0] not in path:  # using path because Queue is not iterable
                # The successor is not in closed so we do want to add it to open
                currentState.currentState = copy.deepcopy(m[0])
                # make it point back to n
                currentState.next1 = n
                currentState.name = name
                name += 1
                # h1 = wrongTilesAmount(currentState.currentState)
                h2 = manhattanDistance(currentState.currentState)
                currentState.heuristic = h2 # h1 + h2
                stateToAdd = copy.deepcopy(currentState)
                open.put((stateToAdd.heuristic, stateToAdd))
                m.pop(0)
            else:
                # check against already existing node
                m.pop(0)


# -----------------------------PROGRAM BEGINS HERE------------------------------------
print("Hello I'm python and I look very simple which is the complicated part!")

# PART 1, read json file and make sure it's a proper file
validJson = False
obj = Puzzle()
while validJson != True:
    jsonFileName = input("Please type the json filename: ")
    jsonData = obj.readJsonFile(jsonFileName)
    if obj.setUp(jsonData) == True:
        print("")  # I'm going to use this for a new line
        obj.displayGrid()
        validJson = True
# END OF PART 1


# PART 2, create the rules for the sliding part
rules = Rules()
# END OF PART 2

# PART 3 backtracking
state = State()
datalist = []
rulesApplied = []
datalist.append(obj.currentState)
stateExamined = []
stateExamined.append(0)
loop = True
totalStatesVisited = 0
print()
print("Backtrack1 is being performed...")
while loop == True:
    if backtrack1(datalist, rules, state, rulesApplied, stateExamined) == state.FAIL:
        print("Failed to find a solution for Depth Bound: ", BOUND)
        print("Visited ", stateExamined[0], " states. Increasing bound...")
        BOUND += 1
        totalStatesVisited += stateExamined[0]
    else:
        totalStatesVisited += stateExamined[0]
        loop = False
printWinningPath(rulesApplied, totalStatesVisited)
obj.displayGrid()
print()
# END OF PART 3

# PART 4 graph search
print("Graph search is being performed...")
graphSearch(obj.currentState, rules, state)
# END OF PART 4

# PART 5 A* algorithm
print()
print("A* is being performed with H1...")
aStarAlgH1(obj.currentState, rules, state)
print()
print("A* is being performed with H2...")
aStarAlgH2(obj.currentState, rules, state)
# What would be better is to combine aStarAlgH1 and H2 to just one algorithm, I separated them because I was using
# one of the copies to try to improve my algorithm, I understand how messy this is. I don't like it either
print("I don't know why I struggle so much with python :)!")