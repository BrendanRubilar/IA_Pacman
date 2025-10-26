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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        #Primero, se puede aprovechar la variable successorGameState para verificar si el agente gana o pierde
        if successorGameState.isWin():
            #print("WinFlag")
            return float('inf') #Si gana, la idea es darle a entender que ese estado es el mejor posible
        if successorGameState.isLose():
            #print("LoseFlag")
            return float('-inf') #Si pierde, la idea es darle a entender que ese estado es el peor posible, no debe hacer eso nuevamente

        EvaluationScore = successorGameState.getScore()
        
        #Luego aprovechando la matriz de comida, se pueden calcular distancias
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(abs(newPos[0] - fx) + abs(newPos[1] - fy) for (fx, fy) in foodList) 
            #Esto provocará que el agente busque la comida más cercana
            EvaluationScore += 15.0 / (minFoodDist + 1.0)
            EvaluationScore -= 2.5 * len(foodList)
            
        #También se consideran las distancias a los poderes para comer fantasmas
        oldCaps = currentGameState.getCapsules()
        newCaps = successorGameState.getCapsules()
        if len(newCaps) < len(oldCaps): #La idea es que si el agente come un poder, se le de una recompensa
            EvaluationScore += 120.0
        if newCaps: #Luego se verifica si hay poderes ceercanos y se incentiva al agente a ir por ellos
            minCapDist = min(abs(newPos[0] - fx) + abs(newPos[1] - fy) for (fx, fy) in newCaps)
            EvaluationScore += 4.0 / (minCapDist + 1.0)
    
        #El agente debe evitar a los fantasmas y perseguirlos si tiene el poder para comerlos
        activeDists = []
        scaredDists = []
        for ghost, scared in zip(newGhostStates, newScaredTimes):
            gpos = ghost.getPosition()
            if gpos is None:
                continue
            dx = abs(newPos[0] - gpos[0])
            dy = abs(newPos[1] - gpos[1])
            d = dx + dy
            if scared > 0:
                scaredDists.append((d, scared))
            else:
                activeDists.append(d)

        #Si Pacman no tiene el poder, debe evitar a los fantasmas
        if activeDists:
            #print(activeDists)
            minActive = min(activeDists)
            safeR = 5
            if minActive <= 1:
                EvaluationScore -= 1000.0 #Si se acerca a ellos, se penaliza mucho.
            elif minActive == 2:
                EvaluationScore -= 50.0 #Si no está tan cerca, pero sigue siendo peligroso, se penaliza pero mucho menos.
            elif minActive <= safeR:
                #Y el ultimo caso, si está entrando al  rango de peligro se le penaliza muy suavemente.
                EvaluationScore -= 3.0 / (minActive + 1.0)

        #Si el agente puede comer a los fantasmas entonces debe perseguirlos
        if scaredDists:
            d, t = min(scaredDists, key=lambda x: x[0])
            #Pero solo debe hacerlo si tiene tiempo, si no, sería un riesgo
            if t > 0:
                EvaluationScore += (6.0 / (d + 1.0)) + min(3.0, t * 0.1)
    
        #El agente no debe quedarse quieto, por lo que se le debe penalizar si lo hace
        if action == Directions.STOP:
            EvaluationScore -= 15.0
    
        #Durante las pruebas me di cuenta que el agente quedaba en movimientos bucleados, así que añadí una 
        #Pequeña penalización si es que el agente decide empezar a volver en sus pasos
        reverse = {
            Directions.NORTH: Directions.SOUTH,
            Directions.SOUTH: Directions.NORTH,
            Directions.EAST:  Directions.WEST,
            Directions.WEST:  Directions.EAST,
            Directions.STOP:  Directions.STOP
        }
        currentDir = currentGameState.getPacmanState().configuration.direction
        # Penaliza acción que es reversa de la dirección actual
        if action == reverse.get(currentDir, Directions.STOP):
            EvaluationScore -= 3.0
            
        return EvaluationScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        #Se inicia la llamada recursiva Minimax desde la raiz, turno de pacman y profundidad 0
        legal = gameState.getLegalActions(0)
        if not legal:
            return Directions.STOP
        _, action = self._minimax(gameState, agentIndex=0, depth=0)
        return action if action is not None else Directions.STOP

    def _minimax(self, state, agentIndex, depth):
        #Si pacman gana, pierde o se alcanza la profundiad máxima se detiene la recursión
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), None

        #Se obtiene la cantidad de agentes en el escenario, pacman siempre es el 0
        numAgents = state.getNumAgents()

        #Para el agente Pacman se debe usar Max
        if agentIndex == 0:
            bestVal, bestAct = float('-inf'), None
            for act in state.getLegalActions(0):
                succ = state.generateSuccessor(0, act)
                val, _ = self._minimax(succ, 1, depth)
                if val > bestVal:
                    bestVal, bestAct = val, act
            return bestVal, bestAct
        #Mientras que para los fantasmas se usa Min
        else:
            bestVal, bestAct = float('inf'), None
            nextAgent = agentIndex + 1
            wrap = (nextAgent == numAgents)
            nextDepth = depth + 1 if wrap else depth
            nextAgent = 0 if wrap else nextAgent

            for act in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, act)
                val, _ = self._minimax(succ, nextAgent, nextDepth)
                if val < bestVal:
                    bestVal, bestAct = val, act
            return bestVal, bestAct

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #Fallback si no hay acciones legales (evita None)
        legal = gameState.getLegalActions(0)
        if not legal:
            return Directions.STOP
        _, action = self._alphabeta(gameState, agentIndex=0, depth=0,
                                    alpha=float('-inf'), beta=float('inf'))
        return action if action is not None else Directions.STOP
    
    def _alphabeta(self, state: GameState, agentIndex: int, depth: int,
                   alpha: float, beta: float):
        #Igual a la implementación MiniMax original, si pacman gana, pierde o se alcanza la profundiad máxima se detiene la recursión
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), None

        numAgents = state.getNumAgents()

        #Función para Pacman, se usa Max
        if agentIndex == 0:
            bestVal, bestAct = float('-inf'), None
            for act in state.getLegalActions(0): #Las acciones se iteran como las entrega la función getLegalActions
                succ = state.generateSuccessor(0, act)
                val, _ = self._alphabeta(succ, 1, depth, alpha, beta)
                if val > bestVal:
                    bestVal, bestAct = val, act
                if bestVal > beta: #Se realiza la poda beta estricta
                    return bestVal, bestAct
                alpha = max(alpha, bestVal)
            return bestVal, bestAct
        else:
            #Función para los fantasmas, se usa Min
            bestVal, bestAct = float('inf'), None
            nextAgent = agentIndex + 1
            wrap = (nextAgent == numAgents)
            nextDepth = depth + 1 if wrap else depth
            nextAgent = 0 if wrap else nextAgent

            for act in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, act)
                val, _ = self._alphabeta(succ, nextAgent, nextDepth, alpha, beta)
                if val < bestVal:
                    bestVal, bestAct = val, act
                if bestVal < alpha: #Se realiza la poda alpha estricta
                    return bestVal, bestAct
                beta = min(beta, bestVal)
            return bestVal, bestAct

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #Dado que no se pide resolver este problema, pero es necesario rellenarlo para ejecutar el problema 5, se utiliza la solción del problema 4 :)
        alphaBetaAgent = AlphaBetaAgent(depth=self.depth, evalFn=self.evaluationFunction.__name__)
        
        return alphaBetaAgent.getAction(gameState)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #Verificar estado del juego, si ganó o perdió se termina la evaluación
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [g.scaredTimer for g in ghostStates]

    EvaluationScore = currentGameState.getScore()

    #La implementación es similar al problema 1
    #Se premia por la cercanía a la comida y se penaliza en base a la cantidad de comida restante, así se incentiva a ir por ella.
    if food:
        dists = [abs(pos[0]-fx) + abs(pos[1]-fy) for (fx, fy) in food]
        dists.sort()
        near = (1.0/(dists[0]+1.0)) + (1.0/(dists[1]+1.0) if len(dists) > 1 else 0.0)
        EvaluationScore += 12.0 * near
        EvaluationScore -= 4.0 * len(food)

    #Si hay poderes cerca, se da una recompensa por ir a comerlas y al igual que con la comida, se penaliza si aún hay poderes restantes
    #así se incentiva al agente a tomar estos poderes y luego cazar a los fantasmas.
    if capsules:
        capDists = [abs(pos[0]-cx) + abs(pos[1]-cy) for (cx, cy) in capsules]
        EvaluationScore += 8.0 / (min(capDists) + 1.0)
        EvaluationScore -= 15.0 * len(capsules)

    #Trabajar con los fantasmas, se guardan sus distancias y sus estados, si están asustados se guarda el tiempo que les queda
    activeD, scaredPairs = [], []
    for g, t in zip(ghostStates, scaredTimes):
        gpos = g.getPosition()
        if gpos is None:
            continue
        d = abs(pos[0]-gpos[0]) + abs(pos[1]-gpos[1])
        if t > 0:
            scaredPairs.append((d, t))
        else:
            activeD.append(d)

    #Así, luego se puede penalizar la cercanía a los fantasmas 
    if activeD:
        dmin = min(activeD)
        if dmin <= 1:
            return float('-inf')   
        elif dmin == 2:
            EvaluationScore -= 120.0
        elif dmin <= 5:
            EvaluationScore -= 10.0 / (dmin + 0.5)

    #Y se incentiva al agente a buscar a los fantasmas asustados
    if scaredPairs:
        d, t = min(scaredPairs, key=lambda x: x[0])
        if t > 0:
            EvaluationScore += 14.0 / (d + 1.0) + min(6.0, 0.2 * t)

    return EvaluationScore

# Abbreviation
better = betterEvaluationFunction
