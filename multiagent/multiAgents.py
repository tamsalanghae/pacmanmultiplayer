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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        instances = []
        score = 0
        ghost = [manhattanDistance(newPos, i.getPosition()) for i in newGhostStates]
        gav = reduce(lambda x, y: x + y, ghost) / len(ghost)
        if not newFood.asList():
            score = 0
        else:
            for x2 in newFood.asList():
                instances.append(manhattanDistance(newPos, x2))
                score = min(instances)
        return successorGameState.getScore() + min(newScaredTimes) + 1 / (score + 0.1) - (1 / (gav + 0.1))


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        def isEnd(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def findMin(state, depth, ghost_index):
            if isEnd(state, depth):
                return self.evaluationFunction(state)
            else:
                value = float('inf')
                for action in state.getLegalActions(ghost_index):
                    if ghost_index == gameState.getNumAgents() - 1:
                        value = min(value, findMax(state.generateSuccessor(ghost_index, action), depth + 1))
                    else:
                        value = min(value,
                                    findMin(state.generateSuccessor(ghost_index, action), depth, ghost_index + 1))
                return value

        def findMax(state, depth):
            if isEnd(state, depth):
                return self.evaluationFunction(state)
            else:
                value = - float('inf')
                for action in state.getLegalActions(0):
                    value = max(value, findMin(state.generateSuccessor(0, action), depth, 1))
                return value

        actionsWithScore = [(action, findMin(gameState.generateSuccessor(0, action), 0, 1)) for action in
                            gameState.getLegalActions(0)]
        actionsWithScore.sort(key=lambda element: element[1], reverse=True)

        return actionsWithScore[0][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        def isEnd(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def findMax(state, depth, alpha, beta):
            if isEnd(state, depth):
                return self.evaluationFunction(state)
            else:
                value = - float('inf')
                for action in state.getLegalActions(0):
                    value = max(value, findMin(state.generateSuccessor(0, action), depth, 1, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value

        def findMin(state, depth, ghost_index, alpha, beta):
            if isEnd(state, depth):
                return self.evaluationFunction(state)
            else:
                value = float('inf')
                for action in state.getLegalActions(ghost_index):
                    if ghost_index == state.getNumAgents() - 1:
                        value = min(value,
                                    findMax(state.generateSuccessor(ghost_index, action), depth + 1, alpha, beta))
                    else:
                        value = min(value,
                                    findMin(state.generateSuccessor(ghost_index, action), depth, ghost_index + 1, alpha,
                                            beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        maxValue = - float('inf')
        alpha = - float('inf')
        beta = float('inf')
        optimalAction = None
        for action in gameState.getLegalActions(0):
            nextValue = findMin(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if nextValue > maxValue:
                maxValue = nextValue
                optimalAction = action
            alpha = max(alpha, maxValue)
        return optimalAction


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
        "*** YOUR CODE HERE ***"

        def isEnd(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def findMax(state, depth):
            if isEnd(state, depth):
                return self.evaluationFunction(state)
            else:
                value = - float('inf')
                for action in state.getLegalActions(0):
                    value = max(value, findAvg(state.generateSuccessor(0, action), depth, 1))
                return value

        def findAvg(state, depth, ghost_index):
            if isEnd(state, depth):
                return self.evaluationFunction(state)
            else:
                value = 0
                for action in state.getLegalActions(ghost_index):
                    if ghost_index == state.getNumAgents() - 1:
                        value = value + findMax(state.generateSuccessor(ghost_index, action), depth + 1)
                    else:
                        value = value + findAvg(state.generateSuccessor(ghost_index, action), depth, ghost_index + 1)
                return value

        maxValue = - float('inf')
        optimalAction = None

        for action in gameState.getLegalActions(0):
            nextValue = findAvg(gameState.generateSuccessor(0, action), 0, 1)
            if nextValue > maxValue:
                maxValue = nextValue
                optimalAction = action
        return optimalAction
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls()

    # if not new ScaredTimes new state is ghost: return lowest value

    "*** YOUR CODE HERE ***"
    ghosts = [ghost.getPosition() for ghost in newGhostStates]
    foods = newFood.asList()

    if not min(newScaredTimes) > 0 and newPos in ghosts:
        return float('-inf')
    if currentGameState.isLose():
        return float('-inf')

    # if not new ScaredTimes new state is ghost: return lowest value

    foodsSortedByDistance = sorted(foods, key=lambda food: util.manhattanDistance(food, newPos))
    ghostsSortedByDistance = sorted(ghosts, key=lambda ghost: util.manhattanDistance(ghost, newPos))

    score = 0

    if util.manhattanDistance(ghostsSortedByDistance[0], newPos) < 3:
        score -= 300
    if util.manhattanDistance(ghostsSortedByDistance[0], newPos) < 2:
        score -= 1000
    if util.manhattanDistance(ghostsSortedByDistance[0], newPos) < 1:
        return float('-inf')

    if len(currentGameState.getCapsules()) < 2:
        score += 100

    if len(foodsSortedByDistance) == 0 or len(ghostsSortedByDistance) == 0:
        score += scoreEvaluationFunction(currentGameState) + 8
    else:
        score += (
                scoreEvaluationFunction(currentGameState) +
                8 / util.manhattanDistance(foodsSortedByDistance[0], newPos) -
                1 / util.manhattanDistance(ghostsSortedByDistance[0], newPos) -
                1 / util.manhattanDistance(ghostsSortedByDistance[-1], newPos)
        )

    return score

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
