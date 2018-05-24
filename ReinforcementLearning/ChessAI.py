import MonteCarlo.Montecarlo_Tree_Search as Monte


class ChessAI :
    def __init__(self,path):
        # self.monte = Monte.Monte()
        self.decision = None
        self.mcts = Monte.MontecarloTreeSearch(path)


    def set(self,Board):
        self.monte.set_state(Board)

    def ask(self,Board):
        if self.monte.first :
            self.set(Board)
            self.monte.first = False
        self.analyze()
        self.refresh(self.decision)
        return self.decision

    def refresh(self,move):
        self.monte.inherit(move)

    def analyze(self):
        self.decision = self.monte.predict()

    def get_MCTS(self,chessBoard):

        nextMove = self.mcts.MCTS(chessBoard)

        return nextMove
    def getNetwork(self):
        return self.mcts.getNetwork