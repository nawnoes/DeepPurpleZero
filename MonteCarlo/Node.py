import math
import random

Cpuct = 5
class Node:


    def __init__(self, parent = None, command=None, policy_Score = 0, color = None): # 부모로 부터 파생 될때, 부모노드의 정보와 커맨드를 부여받음
        self.command = command  # 명령어
        self.color = color # 현재 노드의 색깔 True면 흰색, False 검은색
        self.visit = 0  # 방문횟수
        self.policy_Score = policy_Score # 정책망
        self.N_rollout = 0
        self.W_rollout = 0
        self.N_value = 0
        self.W_value = 0
        self.child = []  # 자식 노드
        self.parent = parent  # 부모노드
        self.array4096 =None
        self.argmaxOfSoftmax = None
        self.bear_Flag = False
        self.lamda = 0.5
        self.n_vl = 0#3 나중에 멀티 프로세싱으로 여러개의 스레드가 트리를 생성할 떼 사용
        self.nextChildIndex=0

    def __del__(self):
        None
    def set_Child(self, child):
        self.on_Flag()
        self.child = child
    def set_Color(self,color):
        self.color = color
    def set_array4096(self,array4096):
        self.array4096 = array4096
    def set_argmaxOfSoftmax(self,argmaxOfSotfmax):
        self.argmaxOfSoftmax = argmaxOfSotfmax
    def is_child(self,anyNode):
        for child in self.child:
            if anyNode == child:
                return True
        return False
    def is_array4096(self):
        if self.array4096 == None:
            return False
        else:
            return True
    def get_W_rollout(self):
        return self.W_rollout
    def get_W_value(self):
        return self.W_value
    def get_array4096(self):
        return self.array4096
    def get_argmaxOfSoftmax(self):
        return self.argmaxOfSoftmax
    def get_visit(self):
        return self.visit
    def get_Command(self):
        return self.command
    def get_Parent(self):
        return self.parent
    def get_Child(self):
        return self.child
    def get_LengthOfChild(self):
        return len(self.child)
    def get_Color(self):
        return self.color
    def get_Flag(self):
        return self.bear_Flag
    def get_N_rollout(self):
        return self.N_rollout
    def get_N_value(self):
        return self.N_value
    def get_NextChildIndex(self):
        return self.nextChildIndex

    def on_Flag(self):
        self.bear_Flag = True
    def off_Flag(self):
        self.bear_Flag = False

    def add_Visit(self,visit):
        self.visit += visit

    def plus_1_NextChildIndex(self):
        self.nextChildIndex += 1

    def add_ChildNode(self, node):
        if self.bear_Flag == False:
            self.on_Flag()
        self.child.append(node)

    def get_Qu(self):
        #win/games + C_puct * policy_Score * ( root( sigma(other child visit) / ( 1 + my visit ) )
        if self.parent ==None:
            score =0
        else:
            score =  self.calc_Q() + self.calc_u()
        return score
    def calc_Q(self):
        if self.color == False: #흑일때 -부호 붙여서 계산
            W_r= -self.W_rollout
            N_r= self.N_rollout
            W_v= -self.W_value
            N_v= self.N_value
        else:
            W_r= self.W_rollout
            N_r= self.N_rollout
            W_v= self.W_value
            N_v= self.N_value
        N_r +=1
        N_v +=1
        q = (1-self.lamda)*(W_r/N_r) + self.lamda*(W_v/N_v)
        return q
    def calc_u(self):
        #
        # if self.N_rollout ==0:
        #     u =0
        # else:
        # u = Cpuct * self.policy_Score * (math.sqrt(self.sum_other_Visit())/(self.N_rollout+1))
        # u = Cpuct * self.policy_Score * (math.sqrt(self.sum_Siblings_Visit()+1) / (self.N_rollout + 1))
        u= Cpuct * self.policy_Score * (math.sqrt(self.sum_Siblings_Visit()) / (self.N_rollout + 1))
        # u = Cpuct * self.policy_Score * (math.sqrt(self.sum_other_Visit()+1) / (self.N_rollout + 1))

        return u

    def sum_other_Visit(self):
        sumAll = self.parent.sum_childVisit()
        return sumAll - self.visit

    def sum_childVisit(self):
        lenth = len(self.child)
        sum = 0
        for i in range(lenth):
            sum += self.child[i].visit
        return sum

    def sum_Siblings_Visit(self):
        # lenth = len(self.child)
        # sum = 0
        # for i in range(lenth):
        #     sum += self.child[i].get_N_rollout()
        return self.parent.get_visit()
    def renewForSelection(self):
        self.N_rollout += self.n_vl
        if self.color == False: #흑의 경우
            self.W_rollout += self.n_vl
        else:
            self.W_rollout -= self.n_vl
    def renewForBackpropagation(self,rolloutResult, valueNetworkResult):
        #rollout의 값 업데이트
        self.N_rollout = self.N_rollout - self.n_vl + 1

        if self.color == False: #흑의 경우
            self.W_rollout = self.W_rollout - self.n_vl +rolloutResult
        else:
            self.W_rollout = self.W_rollout + self.n_vl +rolloutResult
         #가치망의 값 업데이트
        self.W_value += valueNetworkResult
        self.N_value += 1
    def get_bestChild(self):
        #child에서 가장 selectingScore가 최대인 후보를 선택
        lenth = len(self.child)
        index = 0
        max = 0
        candidates = []
        for i in range(lenth):
            if max < self.child[i].calc_selectingScore():
                #점수가 최대값일때 index를 가진다
                max = self.child[i].calc_selectingScore()
                index = i
                candidates.clear()
                candidates.append(i)
            elif max == self.child[i].calc_selectingScore():
                #선택된 값이 최대값과 같다면 후보로 추가
                candidates.append(i)

        if len(candidates) == 1:
            #반복문이 끝났을때 index는 최대값을 가진다
            #print(max)
            return self.child[index] # 최대값 반환
        else : #선택된 것이 없다면 랜덤으로 자식 선택
            try:
                choice = random.choice(candidates)
            except:
                #print(lenth)
                self.child[index]
            #print(max)
            return self.child[choice]

    def should_expand(self, point):
        if self.visit == point:
            return True
        else:
            return False
    def sumChildPolicyScore(self):
        sum = 0
        for child in self.child:
            sum += child.policy_Score
        return sum
    def get_policyDistribution(self):
        scores = []
        sum = self.sumChildPolicyScore()

        for child in self.child:
            score = child.policy_Score /sum
            scores.append(score)
        return scores
    def get_maxVisitedChildIndex(self):
        lenth = len(self.child)
        max = -1
        index = 0
        for i in range(lenth):
            if max < self.child[i].visit:
                max = self.child[i].visit
                #print(max)
                index = i
        return index

    def is_root(self):
        if self.parent:
            return False
        else:
            return True
    def print_SimpleNodeInfo(self):
        print("command: ",self.command ,"visit: ",self.visit," N_rollout: ",self.N_rollout," Q+u: ",self.get_Qu())
    def print_childInfo(self):
        #자식 노드의 정보를 반환
        lenth = len(self.child)
        print("child")
        for i in range(lenth):
            print(i,"> W_rollout:", self.child[i].get_W_rollout(), "W_value:", self.child[i].get_W_value(), " visit",self.child[i].get_visit())
            print("Q+u : ", self.child[i].get_Qu())
            print("ps : ",self.child[i].policy_Score)
            print("q : ",self.child[i].calc_Q())
            print("u : ",self.child[i].calc_u())
            print("N_rollout: ",self.child[i].get_N_rollout()," N_value: ",self.child[i].get_N_value())
            print("move : ", self.child[i].get_Command())

    def trans_result(self, result):
        rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}  # 게임의 끝, ( 백승 = 1, 흑승 = -1, 무승부, 0 )
        return rm[result]