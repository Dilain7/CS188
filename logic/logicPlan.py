# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def sentence1():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"

    """A = A, B = B, C = C 
    o = Or
    a = And 
    b = biconditional / if and only if
    n = not
    i = implies"""

    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    AorB = logic.disjoin(A, B)
    nAbnBoC = ~A % (~B | C)
    nAonBonC = logic.disjoin(~A, ~B, C)
    return logic.conjoin(AorB, nAbnBoC, nAonBonC)

    "*** END YOUR CODE HERE ***"

def sentence2():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    """A = A, B = B, C = C 
        o = Or
        a = And 
        b = biconditional / if and only if
        n = not
        i = implies"""

    "*** END YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')
    CbBoD = C % (B | D)
    AinBanD = A >> (~B & ~D)
    nBanCiA = ~(B & ~C) >> A
    nDiC = ~D >> C
    return logic.conjoin(CbBoD, AinBanD, nBanCiA, nDiC)


def sentence3():
    """Using the symbols PacmanAlive[1], PacmanAlive[0], PacmanBorn[0], and PacmanKilled[0],
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    "*** BEGIN YOUR CODE HERE ***"
    PacmanAlive1 = logic.PropSymbolExpr('PacmanAlive[1]')
    PacmanAlive0 = logic.PropSymbolExpr('PacmanAlive[0]')
    PacmanBorn0 = logic.PropSymbolExpr('PacmanBorn[0]')
    PacmanKilled0 = logic.PropSymbolExpr('PacmanKilled[0]')
    firstClause = PacmanAlive1 % ((PacmanAlive0 & ~PacmanKilled0) | (~PacmanAlive0 & PacmanBorn0))
    secondClause = ~(PacmanAlive0 & PacmanBorn0)
    thirdClause = PacmanBorn0
    return logic.conjoin(firstClause, secondClause, thirdClause)



def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** BEGIN YOUR CODE HERE ***"
    cnfSentence = logic.to_cnf(sentence)
    satSolve = logic.pycoSAT(cnfSentence)
    if str(satSolve) == "False":
        return False
    return(satSolve)

findModel(sentence1())

def atLeastOne(literals):
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    literalsList = []
    for i in range(len(literals)):
        literalsList.append(literals[i])
    return logic.disjoin(literalsList)



def atMostOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    literalsList = []
    for x in range(len(literals)):
        for y in range(x+1, len(literals)):
            if ~x != ~y:
                disjunc = logic.disjoin(~literals[x] , ~literals[y])
                literalsList.append(disjunc)
    return logic.conjoin(literalsList)


def exactlyOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"

    return conjoin(atLeastOne(literals), atMostOne(literals))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, time = parsed
            plan[int(time)] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


def pacmanSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    possibilities = []
    if not walls_grid[x][y+1]:
        possibilities.append( PropSymbolExpr(var_str, x, y+1, t-1)
                            & PropSymbolExpr('South', t-1))
    if not walls_grid[x][y-1]:
        possibilities.append( PropSymbolExpr(var_str, x, y-1, t-1) 
                            & PropSymbolExpr('North', t-1))
    if not walls_grid[x+1][y]:
        possibilities.append( PropSymbolExpr(var_str, x+1, y, t-1) 
                            & PropSymbolExpr('West', t-1))
    if not walls_grid[x-1][y]:
        possibilities.append( PropSymbolExpr(var_str, x-1, y, t-1) 
                            & PropSymbolExpr('East', t-1))

    if not possibilities:
        return None
    
    return PropSymbolExpr(var_str, x, y, t) % disjoin(possibilities)


def pacmanSLAMSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    moved_tm1_possibilities = []
    if not walls_grid[x][y+1]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x, y+1, t-1)
                            & PropSymbolExpr('South', t-1))
    if not walls_grid[x][y-1]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x, y-1, t-1) 
                            & PropSymbolExpr('North', t-1))
    if not walls_grid[x+1][y]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x+1, y, t-1) 
                            & PropSymbolExpr('West', t-1))
    if not walls_grid[x-1][y]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x-1, y, t-1) 
                            & PropSymbolExpr('East', t-1))

    if not moved_tm1_possibilities:
        return None

    moved_tm1_sent = conjoin([~PropSymbolExpr(var_str, x, y, t-1) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_tm1_possibilities)])

    unmoved_tm1_possibilities_aux_exprs = [] # merged variables
    aux_expr_defs = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, t - 1)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, t - 1)
        unmoved_tm1_possibilities_aux_exprs.append(wall_dir_combined_literal)
        aux_expr_defs.append(wall_dir_combined_literal % wall_dir_clause)

    unmoved_tm1_sent = conjoin([
        PropSymbolExpr(var_str, x, y, t-1),
        disjoin(unmoved_tm1_possibilities_aux_exprs)])

    return conjoin([PropSymbolExpr(var_str, x, y, t) % disjoin([moved_tm1_sent, unmoved_tm1_sent])] + aux_expr_defs)


def pacphysics_axioms(t, all_coords, non_outer_wall_coords):
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
    """
    pacphysics_sentences = []

    "*** BEGIN YOUR CODE HERE ***"
    t = t
    allCoords = all_coords
    legalSquares = non_outer_wall_coords

    wallNotPac = []
    for x , y in allCoords:
        PacmanNotAtX = logic.PropSymbolExpr(wall_str, x, y) >> ~logic.PropSymbolExpr(pacman_str, x, y, t)
        wallNotPac.append(PacmanNotAtX)

    wallIn = []
    for x, y in legalSquares:
        pacIN = logic.PropSymbolExpr(pacman_str, x, y, t)
        wallIn.append(pacIN)

    pacphysics_sentences.append(conjoin(wallNotPac))
    pacphysics_sentences.append(exactlyOne(wallIn))

    pacActions = []
    for x in DIRECTIONS:
        pacsDirec = logic.PropSymbolExpr(x, t)
        pacActions.append(pacsDirec)

    pacphysics_sentences.append(exactlyOne(pacActions))

    return conjoin(pacphysics_sentences)


def check_location_satisfiability(x1_y1, x0_y0, action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - problem = An instance of logicAgents.LocMapProblem
    Return:
        - a model proving whether Pacman is at (x1, y1) at time t = 1
        - a model proving whether Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** BEGIN YOUR CODE HERE ***"
    #at time 0
    KB.append(logic.PropSymbolExpr(pacman_str, x0, y0, 0))
    KB.append(pacphysics_axioms(0, all_coords, non_outer_wall_coords))
    KB.append(logic.PropSymbolExpr(action0, 0))
    KB.append(allLegalSuccessorAxioms(1, walls_grid, non_outer_wall_coords))
    #at time 1
    KB.append(pacphysics_axioms(1, all_coords, non_outer_wall_coords))
    KB.append(logic.PropSymbolExpr(action1, 1))


    Q = PropSymbolExpr(pacman_str, x1, y1, 1)
    KBanQ = conjoin(KB) & ~Q
    KBaQ = conjoin(KB) & Q


    model1 = findModel(KBanQ)
    model2 = findModel(KBaQ)

    return(model1, model2)



def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str, x0, y0, 0))


    for t in range(50):
        nonWallPac = []
        for x, y in non_wall_coords:
            nonWallPac.append(PropSymbolExpr(pacman_str, x, y, t))
        KB.append(exactlyOne(nonWallPac))

        goalAssertion = PropSymbolExpr(pacman_str, xg, yg, t)

        potentialModel = findModel((conjoin(KB) & goalAssertion))

        if potentialModel:
            return extractActionSequence(potentialModel, actions)
        oneAction = []
        for a in actions:
            oneAction.append(PropSymbolExpr(a, t))
        KB.append(exactlyOne(oneAction))

        transitionM = []
        for x, y in non_wall_coords:
            transitionM.append(pacmanSuccessorStateAxioms(x, y, t+1, walls, var_str=pacman_str))
        KB.append(conjoin(transitionM))


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    #locations = list(filter(lambda loc : loc not in walls_list, all_coords))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str, x0, y0, 0))
    initializeFood = []
    for x, y in food:
        initializeFood.append(PropSymbolExpr(food_str, x, y, 0))
    KB.append(conjoin(initializeFood))

    for t in range(50):
        print("Kobe Bean at timestep: " + str(t))
        nonWallPac = []
        for x, y in non_wall_coords:
            nonWallPac.append(PropSymbolExpr(pacman_str, x, y, t))
        KB.append(exactlyOne(nonWallPac))

        goalAssertion = []
        for x, y in food:
            goalAssertion.append(~PropSymbolExpr(food_str, x, y, t))
        print("KOBE" + str(t))
        potentialModel = findModel((conjoin(KB) & conjoin(goalAssertion)))
        print("LEBRON", str(t))
        if potentialModel:
            return extractActionSequence(potentialModel, actions)

        oneAction = []
        for a in actions:
            oneAction.append(PropSymbolExpr(a, t))
        KB.append(exactlyOne(oneAction))

        foodSucc = []
        for x, y in food:
            foodAtT = PropSymbolExpr(food_str, x, y, t)
            foodAtT1 = PropSymbolExpr(food_str, x, y, t+1)
            pacmanFAtT = PropSymbolExpr(pacman_str, x, y, t)
            faP = (foodAtT & pacmanFAtT) >> ~foodAtT1
            fanP = (foodAtT & ~pacmanFAtT) >> foodAtT1
            # nfaP = (~foodAtT & pacmanAtT) >> ~foodAtT1
            foodSucc.append(faP)
            foodSucc.append(fanP)
            # foodSucc.append(nfaP)

        KB.append(conjoin(foodSucc))

        transitionM = []
        for x, y in non_wall_coords:
            transitionM.append(pacmanSuccessorStateAxioms(x, y, t + 1, walls, var_str=pacman_str))
        KB.append(conjoin(transitionM))

# Helpful Debug Method
def visualize_coords(coords_list, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualize_bool_array(bool_arr, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)


def sensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def four_bit_percept_rules(t, percepts):
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def num_adj_walls_percept_rules(t, percepts):
    """
    SLAM uses a weaker num_adj_walls sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    num_adj_walls = sum(percepts)
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t, walls_grid, non_outer_wall_coords): 
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t, walls_grid, non_outer_wall_coords): 
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSLAMSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def localization(problem, agent):
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    debug = False

    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    possible_locs_by_timestep = []
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    wallCheck = []
    for x,y in all_coords:
        if (x,y) in walls_list:
            wallCheck.append(PropSymbolExpr(wall_str, x, y))
        else:
            wallCheck.append(~PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(wallCheck))


    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], t))

        KB.append(sensorAxioms(t, non_outer_wall_coords))
        percepts = agent.getPercepts()
        KB.append(four_bit_percept_rules(t, percepts))

        possible_locations_t = []
        for x , y in non_outer_wall_coords:
            pacmanExist = PropSymbolExpr(pacman_str, x, y, t)
            pacProveE = findModel(conjoin(KB) & ~pacmanExist)
            pacProveNE = findModel(conjoin(KB) & pacmanExist)

            if not pacProveE:
                KB.append(pacmanExist)

            if not pacProveNE:
                KB.append(~pacmanExist)

            if pacProveNE:
                possible_locations_t.append((x,y))

        possible_locs_by_timestep.append(possible_locations_t)
        agent.moveToNextState(agent.actions[t])
        KB.append(allLegalSuccessorAxioms(t+1, walls_grid, non_outer_wall_coords))

    "*** END YOUR CODE HERE ***"
    return possible_locs_by_timestep


def mapping(problem, agent):
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    #map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]
    known_map_by_timestep = []

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, 0))

    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], t))

        KB.append(sensorAxioms(t, non_outer_wall_coords))
        percepts = agent.getPercepts()
        KB.append(four_bit_percept_rules(t, percepts))

        for x, y in non_outer_wall_coords:
            wallExist = PropSymbolExpr(wall_str, x, y)
            wallProve = findModel(conjoin(KB) & ~wallExist)
            wallNProve = findModel(conjoin(KB) & wallExist)

            if not wallProve:
                KB.append(wallExist)
                known_map[x][y] = 1

            if not wallNProve:
                KB.append(~wallExist)
                known_map[x][y] = 0

        known_map_by_timestep.append(copy.deepcopy(known_map))
        agent.moveToNextState(agent.actions[t])
        KB.append(allLegalSuccessorAxioms(t+1, known_map, non_outer_wall_coords))


    return known_map_by_timestep


def slam(problem, agent):
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]
    known_map_by_timestep = []
    possible_locs_by_timestep = []

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, 0))

    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], t))

        KB.append(SLAMSensorAxioms(t, non_outer_wall_coords))
        percepts = agent.getPercepts()
        KB.append(num_adj_walls_percept_rules(t, percepts))

        for x, y in non_outer_wall_coords:
            wallExist = PropSymbolExpr(wall_str, x, y)
            wallProve = findModel(conjoin(KB) & ~wallExist)
            wallNProve = findModel(conjoin(KB) & wallExist)

            if not wallProve:
                KB.append(wallExist)
                known_map[x][y] = 1

            if not wallNProve:
                KB.append(~wallExist)
                known_map[x][y] = 0

        known_map_by_timestep.append(copy.deepcopy(known_map))

        possible_locations_t = []
        for x, y in non_outer_wall_coords:
            pacmanExist = PropSymbolExpr(pacman_str, x, y, t)
            pacProveE = findModel(conjoin(KB) & ~pacmanExist)
            pacProveNE = findModel(conjoin(KB) & pacmanExist)

            if not pacProveE:
                KB.append(pacmanExist)

            if not pacProveNE:
                KB.append(~pacmanExist)

            if pacProveNE:
                possible_locations_t.append((x, y))

        possible_locs_by_timestep.append(possible_locations_t)
        agent.moveToNextState(agent.actions[t])

        known_map_update = copy.deepcopy(known_map)
        for x in range(problem.getWidth() +2):
            for y in range(problem.getHeight()+2):
                if known_map[x][y] == 1:
                    known_map_update[x][y] = True
                else:
                    known_map_update[x][y] = False

        KB.append(SLAMSuccessorAxioms(t+1, known_map_update, non_outer_wall_coords))

    return known_map_by_timestep, possible_locs_by_timestep

# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)