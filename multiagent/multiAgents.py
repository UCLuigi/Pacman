# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldFood = currentGameState.getFood().asList()
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        capsules = currentGameState.getCapsules()

        returnValue = 0

        for capsule in capsules:
            if newPos == capsule:
                returnValue += 12

        for aFoodItem in oldFood:
            if newPos == aFoodItem:
                returnValue += 6
            else:
                md = float(manhattanDistance(newPos, aFoodItem))
                returnValue += 1 / md

        for daGhost in newGhostStates:
            if newPos == daGhost.getPosition() and daGhost.scaredTimer > 0:
                returnValue += 150
            elif newPos == daGhost.getPosition() and daGhost.scaredTimer <= 0:
                returnValue -= 250
            elif manhattanDistance(newPos, daGhost.getPosition()) <= 1 and 2 > daGhost.scaredTimer > 0:
                returnValue += 2
            elif manhattanDistance(newPos, daGhost.getPosition()) <= 2 and 3 > daGhost.scaredTimer > 0:
                returnValue += 1
            elif manhattanDistance(newPos, daGhost.getPosition()) <= 3 and 4 > daGhost.scaredTimer > 0:
                returnValue += .5
            elif manhattanDistance(newPos, daGhost.getPosition()) <= 3 and daGhost.scaredTimer <= 0:
                returnValue -= 3 * manhattanDistance(newPos, daGhost.getPosition())

        return returnValue


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth):
            if state.isLose() or state.isWin() or depth + 1 == self.depth:
                return self.evaluationFunction(state)
            v = float("-inf")
            for move in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, move), 1, depth + 1))
            return v

        def min_value(state, agent, depth):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            v = float("inf")
            for move in state.getLegalActions(agent):
                if agent == state.getNumAgents() - 1:
                    v = min(v, max_value(state.generateSuccessor(agent, move), depth))
                else:
                    v = min(v, min_value(state.generateSuccessor(agent, move), agent + 1, depth))
            return v

        bestaction = Directions.RIGHT
        score = float("-inf")
        for action in gameState.getLegalActions(0):
            currscore = min_value(gameState.generateSuccessor(0, action), 1, 0)
            if currscore > score:
                score = currscore
                bestaction = action
        return bestaction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, a, b, depth):
            if state.isLose() or state.isWin() or depth + 1 == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for move in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, move), a, b, 1, depth + 1))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def min_value(state, a, b, agent, depth):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            v = float('inf')
            for move in state.getLegalActions(agent):
                if state.getNumAgents() - 1 == agent:
                    v = min(v, max_value(state.generateSuccessor(agent, move), a, b, depth))
                else:
                    v = min(v, min_value(state.generateSuccessor(agent, move), a, b, agent + 1, depth))
                if v < a:
                    return v
                b = min(b, v)
            return v

        bestaction = Directions.RIGHT
        score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            currscore = min_value(gameState.generateSuccessor(0, action), alpha, beta, 1, 0)
            if currscore > score:
                score = currscore
                bestaction = action
            if currscore > beta:
                return bestaction
            alpha = max(alpha, currscore)
        return bestaction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        def max_val(state, depth):
            if state.isLose() or state.isWin() or depth + 1 == self.depth:
                return float(self.evaluationFunction(state))
            v = float('-inf')
            for move in state.getLegalActions(0):
                v = max(v, exp_val(state.generateSuccessor(0, move), 1, depth + 1))
            return v

        def exp_val(state, agent, depth):
            if state.isWin() or state.isLose():
                return float(self.evaluationFunction(state))
            v = float(0)
            for move in state.getLegalActions(agent):
                if agent == state.getNumAgents() - 1:
                    v += max_val(state.generateSuccessor(agent, move), depth)
                else:
                    v += exp_val(state.generateSuccessor(agent, move), agent + 1, depth)
            return v / float(len(state.getLegalActions(agent)))

        score = float('-inf')
        bestaction = Directions.STOP
        for action in gameState.getLegalActions(0):
            currscore = exp_val(gameState.generateSuccessor(0, action), 1, 0)
            if currscore > score:
                score = currscore
                bestaction = action
        return bestaction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: So this is pretty much the same process as the first evaluation function, however a few things have
      changed. We don't really have to give points for actually getting the food, because we want to get to the next
      state that gets the food. We used the inverse for the mh distance.

      And for the boost if you reach the game state - You win the game if the game state is in the winning state, so
      that's why you give it to he most points.
    """
    food = currentGameState.getFood().asList()
    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    if (position == (15, 2)):
        test = 0
    returnValue = 0
    if currentGameState.isWin():
        returnValue += 4000
    for capsule in capsules:
        if position == capsule:
            returnValue += 12

    for aFoodItem in food:
        if position != aFoodItem:
            md = float(manhattanDistance(position, aFoodItem))
            returnValue += 4 / md

    for daGhost in ghostStates:
        if position == daGhost.getPosition() and daGhost.scaredTimer > 0:
            returnValue += 150
        elif position == daGhost.getPosition() and daGhost.scaredTimer <= 0:
            returnValue -= 400
        elif manhattanDistance(position, daGhost.getPosition()) <= 1 and 2 > daGhost.scaredTimer > 0:
            returnValue += 2
        elif manhattanDistance(position, daGhost.getPosition()) <= 2 and 3 > daGhost.scaredTimer > 0:
            returnValue += 1
        elif manhattanDistance(position, daGhost.getPosition()) <= 3 and 4 > daGhost.scaredTimer > 0:
            returnValue += .5
        elif manhattanDistance(position, daGhost.getPosition()) <= 3 and daGhost.scaredTimer <= 0:
            returnValue -= 4 * manhattanDistance(position, daGhost.getPosition())

    return returnValue + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

