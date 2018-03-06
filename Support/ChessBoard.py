import chess

class PreMove:
    def __init__(self,num=5):
        self.moves =[]
        self.MAX = 5
    def lenOfMoves(self):
        return len(self.moves)
    def push(self,move):
        if self.lenOfMoves() < self.MAX:
            self.moves.append(move)
        else:
            self.pop()
            self.moves.append(move)
    def pop(self):
            self.moves.pop(0) #가장 오래된 move 삭제
    def clear(self):
        while not self.is_empty():
            self.pop()
    def is_empty(self):
        if self.lenOfMoves() == 0 :
            return True
        else:
            return False
    def returnPremoves(self):
        mov = []
        lenOfMoves = self.lenOfMoves()
        differ = self.MAX - lenOfMoves
        k=0
        while lenOfMoves > 0 :
            mov.append(self.moves[lenOfMoves - 1])
            lenOfMoves -= 1
        for i in range(differ):
            mov.append('None')
        return mov
    def returnStringPremoves(self):

        lenOfMoves = self.lenOfMoves()
        differ = self.MAX - lenOfMoves
        k=0
        str =""
        while lenOfMoves > 0 :
            str += self.moves[lenOfMoves - 1]+":"
            lenOfMoves -= 1
        for i in range(differ):
            str += 'None:'
        return str
class ChessBoard:
    def __init__(self):
        self.chessBoard = chess.Board()
        self.preMoves = PreMove()
    def push(self,stringMove):
        #push가 된 시점에서 학습을 위한 Data를 반환?
        fen = self.chessBoard.fen()
        data = self.makeDataForRL(fen,stringMove,self.preMoves)
        self.chessBoard.push(chess.Move.from_uci(stringMove))
        self.preMoves.push(stringMove)
        return data
    def getLegalMoves(self):
        return self.chessBoard.legal_moves
    def getTurn(self):
        return self.chessBoard.turn
    def getChessBoard(self):
        return self.chessBoard
    def getFen(self):
        return self.chessBoard.fen()
    def getPremoves(self):
        preMoves = self.preMoves.returnPremoves()
        return preMoves
    def getStringPremoves(self):
        return self.preMoves.returnStringPremoves()
    def makeDataForRL(self,fen,stringMove,moves):
        preMoves = moves.returnPremoves()
        str =fen + ":" +stringMove+":"+preMoves[0]+":"+preMoves[1]+":"+preMoves[2]+":"+preMoves[3]+":"+preMoves[4]+":"+"*"
        return str
    def getResult(self):
        return self.chessBoard.result()
    def is_game_over(self):
        return self.chessBoard.is_game_over()
    def getData(self):
        return self.getFen(), self.getPremoves()