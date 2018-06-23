import json

class GameInfo:
    def __init__(self):
        self.info = {
            'GameCount': 0,
            'CurrentGameCount': 0,
            'RotationCount' : 0,
            'FormerWin': 0,
            'LaterWin': 0,

            'FormerBlackWin': 0,
            'FormerWhiteWin': 0,

            'LaterBlackWin': 0,
            'LaterWhiteWin': 0,

            'CurrentLaterBlackWin': 0,
            'CurrentLaterWhiteWin': 0,
            'CurrentFormerBlackWin': 0,
            'CurrentFormerWhiteWin': 0
        }
        self.savePath="../Data/Self-Play_info.json"

    def load(self):
        try:
            with open(self.savePath,mode='r') as f:
                self.info = json.loads(json.load(f))
                print(self.info)
                print(type(self.info))

        except:
            with open(self.savePath,mode='w') as f:
                json_string= json.dumps(self.info)
                json.dump(json_string,f,indent=4)
    def save(self):
        json_string = json.dumps(self.info)
        with open(self.savePath, mode='w') as f:
            json.dump(json_string, f, indent=4)
    def get_ProbOfCLW(self):
        prob= (((self.info['CurrentLaterBlackWin']+
               self.info['CurrentLaterWhiteWin'])/
              self.info['CurrentGameCount']) * 100)
        return prob
    def is_ChangeCheckpoint(self):
        prob = self.get_ProbOfCLW()
        if self.info['CurrentGameCount'] > 20 and prob>50:
            return True
        else:
            return False

    def initilizeInfo(self):
        self.info['CurrentGameCount']=0
        self.info['CurrentFormerWhiteWin']=0
        self.info['CurrentFormerBlackWin']=0
        self.info['CurrentLaterWhiteWin']=0
        self.info['CurrentLaterBlackWin']=0
    def upRotaionCount(self):
        self.info['RotationCount']+=1


