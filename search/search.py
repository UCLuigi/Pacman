# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    #Key (X, Y) coordinate, Value = ((x,y), action). Value is what came before it, and what action it took to reach this from before
    dictionary = dict()
    newStack = util.Stack() #holds coordinates, actions, and cost. First in last out
    visited = set() #holds coordinates
    #Visited First Starting Node Already, adding the coordinates
    visited.add(problem.getStartState())
    #This adds the first neighbor in the Stack first
    returnValue = []
    for beginningChildren in problem.getSuccessors(problem.getStartState()):
        bcCoordinates = beginningChildren[0]
        bcActions = beginningChildren[1]
        newStack.push(beginningChildren)
        dictionary[bcCoordinates] = (problem.getStartState(), bcActions)

    while not newStack.isEmpty():
        node = newStack.pop() #The node is saved as (x, y), actions, and costs
        coordinates = node[0]
        action = node[1]
        if coordinates not in visited:
            visited.add(coordinates)
        x = coordinates
        if problem.isGoalState(coordinates):
            while (x != problem.getStartState()):
                previousCoordinate, previousAction = dictionary[x]
                returnValue.append(previousAction)
                x = previousCoordinate
            returnValue.reverse()
            return returnValue
        else:
            childrenNodes = problem.getSuccessors(coordinates)  # Each is saved as 0: (x, y), action (South), cost
            for nodesInfo in childrenNodes:
                nodesInfoCoordinates = nodesInfo[0]
                nodesInfoActions = nodesInfo[1]
                if nodesInfoCoordinates not in visited:
                    newStack.push(nodesInfo)
                    dictionary[nodesInfoCoordinates] = (coordinates, nodesInfoActions)

    return returnValue


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    newStack = util.Queue() #holds coordinates, actions, and cost. First in last out
    visited = set() #holds coordinates
    #Visited First Starting Node Already, adding the coordinates
    #This adds the first neighbor in the Stack first
    returnValue = []
    newStack.push((problem.getStartState(), returnValue))


    while not newStack.isEmpty():
        node = newStack.pop() #The node is saved as (x, y)
        coordinates = node[0]
        currentPath = node[1]
        if problem.isGoalState(coordinates):
            return currentPath
        elif coordinates not in visited:
            visited.add(coordinates)
            childrenNodes = problem.getSuccessors(coordinates)  # Each is saved as 0: (x, y), action (South), cost
            for nodesInfo in childrenNodes:
                nodesInfoCoordinates = nodesInfo[0]
                nodesInfoActions = currentPath + [nodesInfo[1]]
                newStack.push((nodesInfoCoordinates, nodesInfoActions))


    return returnValue

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    newStack = util.PriorityQueue() #holds coordinates, actions, and cost. First in last out
    visited = set() #holds coordinates
    #Visited First Starting Node Already, adding the coordinates
    #This adds the first neighbor in the Stack first
    returnValue = []
    newStack.push((problem.getStartState(), returnValue, 0), 0)


    while not newStack.isEmpty():
        node = newStack.pop() #The node is saved as (x, y)
        coordinates = node[0]
        currentPath = node[1]
        currentCost = node[2]
        if problem.isGoalState(coordinates):
            return currentPath
        elif coordinates not in visited:
            visited.add(coordinates)
            childrenNodes = problem.getSuccessors(coordinates)  # Each is saved as 0: (x, y), action (South), cost
            for nodesInfo in childrenNodes:
                nodesInfoCoordinates = nodesInfo[0]
                nodesInfoActions = currentPath + [nodesInfo[1]]
                nodesInfoCosts = nodesInfo[2] + currentCost
                newStack.push((nodesInfoCoordinates, nodesInfoActions, nodesInfoCosts), nodesInfoCosts)


    return returnValue

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    newStack = util.PriorityQueue() #holds coordinates, actions, and cost. First in last out
    visited = set() #holds coordinates
    #Visited First Starting Node Already, adding the coordinates
    #This adds the first neighbor in the Stack first
    returnValue = []
    newStack.push((problem.getStartState(), returnValue, 0), 0)


    while not newStack.isEmpty():
        node = newStack.pop() #The node is saved as (x, y)
        coordinates = node[0]
        currentPath = node[1]
        currentCost = node[2]
        if problem.isGoalState(coordinates):
            return currentPath
        elif coordinates not in visited:
            visited.add(coordinates)
            childrenNodes = problem.getSuccessors(coordinates)  # Each is saved as 0: (x, y), action (South), cost
            for nodesInfo in childrenNodes:
                nodesInfoCoordinates = nodesInfo[0]
                nodesInfoActions = currentPath + [nodesInfo[1]]
                nodesInfoCosts = nodesInfo[2] + currentCost
                nodesInfoCostsAndHeuristic = heuristic(nodesInfoCoordinates, problem) + nodesInfoCosts
                newStack.push((nodesInfoCoordinates, nodesInfoActions, nodesInfoCosts), nodesInfoCostsAndHeuristic)


    return returnValue


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
