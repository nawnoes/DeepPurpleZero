from numba import jit
from numpy import arange
from Support.Board2Array import Board2Array as B2A
import chess
import time
from ReinforcementLearning.ChessAI import ChessAI
# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.

@jit
def get_JITinput(chessBoard):
    start = time.time()
    # ca = ChessAI('../Checkpoint/Later/')
    # ca.get_MCTS(chessBoard)
    b2a=B2A()
    b2a.board2array(chessBoard)
    end = time.time()

    elapsed = end - start
    print(elapsed)
    return elapsed
def get_input(chessBoard):
    start = time.time()
    # ca = ChessAI('../Checkpoint/Later/')
    # ca.get_MCTS(chessBoard)
    b2a = B2A()
    b2a.board2array(chessBoard)
    end = time.time()

    elapsed = end - start
    print(elapsed)
    return elapsed

cb= chess.Board()
# get_input(cb)
sum = 0
for i in range(100):
    # get_JITinput(cb)
    sum += get_JITinput(cb)

print(sum)