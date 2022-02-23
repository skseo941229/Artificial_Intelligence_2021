import sys
import getopt
import numpy as np
from decimal import Decimal, getcontext
import random
def build_matrix(rows, cols):
    matrix = []
    for r in range(0, rows):
        matrix.append([0 for c in range(0, cols)])
    return matrix
def build_inside(matrix, actions, states,probabilities, policy):
    for keys in actions:
        total = 1; 
        if keys not in probabilities and len(actions[keys]) !=1: ##edges는 여러개지만 probabilities는 없음, 그래서 probability1개
            matrix[states.index(keys)][states.index(policy[keys])] = 1
        #elif len(probabilities[keys]) == 1 and probabilities[keys][0] == 1:
         #   print(keys, "2")
          #  num = len(actions[keys])
          #  val = 1/num
          #  for items in actions[keys]:
        #     matrix[states.index(keys)][states.index(items)] = 0
        elif keys not in probabilities and len(actions[keys]) ==1:  ##edges도 한개 probabilities는 없음, 그래서 probability1개
            matrix[states.index(keys)][states.index(actions[keys][0])] = 1
        elif len(actions[keys]) > 1 and probabilities[keys][0] == 1: ## edges는 여러개지만, probabilities =1이 하나 
            matrix[states.index(keys)][states.index(policy[keys])] = 1
        
        elif len(probabilities[keys]) == 1: ##
            a = states.index(policy[keys])
            matrix[states.index(keys)][a] = probabilities[keys][0]
            total =1- probabilities[keys][0]
            
            num = len(actions[keys])-1
            val= total/num
            
            for items in actions[keys]:
                if items not in policy[keys]:
                    matrix[states.index(keys)][states.index(items)] = val
        elif len(probabilities[keys]) == len(actions[keys]):
            start =0
            for i in range(len(actions[keys])):
                a = states.index(actions[keys][i])
                matrix[states.index(keys)][a] = probabilities[keys][i]
                start = start+1  
    return matrix

def main():
  try:
    opts, args = getopt.getopt(sys.argv[1:],"dmti", ["df=","min","tol=", "iter="])
  except getopt.GetoptError as err:
    print (err)
    sys.exit(1)
  tolerance = 0.01
  iteration = 100
  min_flag = 0
  discount_factor = 1
    

  for opt,arg in opts:
    if ( opt == "-d" ) or (opt == "--df"):
        discount_factor = float(arg)
    elif ( opt == "-m" ) or ( opt == "--min"):
        min_flag =1
    elif ( opt == "-t" ) or ( opt == "--tol"):
        tolerance = float(arg)
    elif ( opt == "-i" ) or( opt == "--iter" ):
        iteration = int(arg)

  filename = sys.argv[len(sys.argv)-1]
  f = open(filename, 'r')
  data = []
  while True:
    line = f.readline()
    if not line: break
    line = line.strip()
    data.append(line)
  f.close()
  data_rewards = []
  data_probabilities = []
  data_goto = []

  for i in range(len(data)):
    if  data[i] =='':
        continue    
    elif data[i][0] == "#":
        continue 
    elif data[i].find("=") != -1 :
        data_rewards.append(data[i])
    elif data[i].find("%") != -1 : 
        data_probabilities.append(data[i])
    elif data[i].find(":") != -1 :
        data_goto.append(data[i])
  actions = dict()
  for j in range(len(data_goto)):
    mid1 = data_goto[j].replace('[','')
    mid2 = mid1.replace(']','')
    mid3 = mid2.replace(':',',')
    mid4 = mid3.split(',')
    remove_space = [x.strip(' ') for x in mid4]
    reslist = []
    for i in range(len(remove_space)):
        reslist.append(remove_space[i+1])
        if i+1== len(remove_space)-1: break
    actions[remove_space[0]] = reslist
  states = []
  for keys in actions:
    if keys not in states:
        states.append(keys)
    for items in actions[keys]:
         if items not in states:
            states.append(items)  
  states.sort()
  rewards={}
  for i in range(len(data_rewards)):
    mid= data_rewards[i].split("=",1)
    try:
        int(mid[1])
    except ValueError:
        print ("Check Reward Value! In this example, only integer type is allowed in Reward")
        return 
    rewards[mid[0]] = int(mid[1])

  for i in range(len(states)) :
    if states[i] not in rewards:
        rewards[states[i]]=0
  probabilities={}
  for i in range(len(data_probabilities)):
    mid= data_probabilities[i].split("%",1)
    remove_space = [x.strip(' ') for x in mid]
    problist = remove_space[1].split(' ')
    reslist = []
    for i in range(len(problist)):
        try:
            float(problist[i])
        except ValueError:
            print ("Please check probability entries, invalid input")
            return 
        val = float(problist[i])
        reslist.append(val)
    #reslist.sort()
    probabilities[remove_space[0]] = reslist
  policy={}
  for keys in actions:
    if keys in probabilities :
        if len(actions[keys]) > len(probabilities[keys]):
            policy[keys] = np.random.choice(actions[keys])
    else :        
        if len(actions[keys]) > 1:
            policy[keys] = np.random.choice(actions[keys])
  Value = {}
  for sts in states:
        if sts in rewards.keys():
            Value[sts] = rewards[sts]
        else:
            Value[sts] = 0
  size_matrix  = len(states) 
  matrix = build_matrix(size_matrix , size_matrix)
  matrix = build_inside(matrix, actions, states,probabilities, policy)
  cur_iteration =0
  biggest_change = 0
  while 1:
    cur_iteration +=1
    
    matrix = build_matrix(size_matrix , size_matrix)
    matrix = build_inside(matrix, actions, states,probabilities, policy)
    for sts in states:
                a = states.index(sts)
                new_val = rewards[sts]
                val =0
                old_val = Value[sts]
                for dest in states: 
                    b= states.index(dest)
                    vval = discount_factor*matrix[a][b]*Value[dest]
                    new_val += vval
                biggest_change = max(biggest_change, abs(old_val-new_val))
                Value[sts] = new_val
    if biggest_change <tolerance:
        break

    for sts in states:  
        a = states.index(sts)
        new_val = rewards[sts]
        maxval =0
        minval = 10000
        if sts in actions:
            for dest in actions[sts]: 
                b= states.index(dest)
                vval = Value[dest]
                if vval > maxval and sts in policy and min_flag ==0:
                    maxval = vval
                    policy[sts] = dest
                elif vval < minval and sts in policy and min_flag ==1:
                    minval = vval
                    policy[sts] = dest
    if cur_iteration == iteration:
        break
  for keys in policy:
    print(keys, "->", policy[keys])
  for vals in Value:
    print(vals, "=", Value[vals])
  return
 
if __name__ == '__main__':
  main()

