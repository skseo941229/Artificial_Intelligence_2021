import sys
import getopt
import numpy as np
from decimal import Decimal, getcontext
import random
import argparse
import pandas as pd
from collections import Counter
def kmeans_algo(initial_point, data, distance_method,keys):
    while(1):
        prev_point =[]
        dist_res = {}
        prev_point = initial_point[:]
        for row in keys:
            res = []
            for i in range(len(initial_point)):
                if distance_method ==1:
                    dist = euclidean_dist(data[row], initial_point[i])
                else:
                    dist = manhattan(data[row], initial_point[i])
                    ##print(initial_point[i], row, dist)
                res.append(dist)
            new_key = np.argmin(res)
            if new_key in dist_res.keys():
                dist_res[new_key].append(row)
            else:
                dist_res[new_key] = []
                dist_res[new_key].append(row)
        for i in dist_res.keys():
            sum = 0
            cnt =0
            for col in dist_res[i]:
                sum += data[col]
                cnt +=1
            initial_point[i] = list(sum/cnt)
        if  np.array_equal(prev_point,initial_point) ==True:
            break
    return dist_res, initial_point

def euclidean_dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))
def most_common(lst):
    return max(set(lst), key=lst.count)
def unit_vote (dist, n_cluster):
    inside=[]
    for i in range(n_cluster):
        inside.append(dist[i][1])
    occurence_count = Counter(inside)
    tmp = occurence_count.most_common(1)[0][0]
    #print(tmp)
    return tmp
def distance_vote(dist, n_cluster):
    revec = {}
    for i in range(n_cluster):
        if dist[i][0] ==0:
            dist[i][0]=1
        if dist[i][1] in revec.keys():
            revec[dist[i][1]] += (1/dist[i][0])
        else:
            revec[dist[i][1]] =0
            revec[dist[i][1]] += (1/dist[i][0])
        
    tmp= max(revec, key=revec.get)
    return tmp   
def main():
  try:
    opts, args = getopt.getopt(sys.argv[2:],"k:u", ["dist=","k=","unitw","train=","test=", "data="])
  except getopt.GetoptError as err:
    print (err)
    sys.exit(1)

  n_cluster = 3 
  dist_method = 1 #euclidean #0-> manh
  voting = 0 # not set - 1/d 
  mode = 0
  train_data=""
  test_data=""
  input_data=""
  here =0

  if sys.argv[1] =='KNN' or sys.argv[1]=='knn' or sys.argv[1]=='kNN':
    mode =0
  else:
    mode =1 
    
  for opt,arg in opts:
    if ( opt == "--dist"):
        if arg == 'manh':
            dist_method = 0
    elif ( opt == "-k" ) or ( opt == "--k"):
        n_cluster = int(arg)
    elif ( opt == "-u" ) or ( opt == "--unitw"):
        voting = 1
    elif ( opt == "--train" ):
        train_data = arg
    elif ( opt == "--test" ):
        test_data = arg
    elif ( opt == "--data" ):
        #here = 
        input_data = arg
  
  initial_point1=[]
  if mode == 1:
    tmp1 = sys.argv[len(sys.argv)-1].split(',')
    initial_point1.append(tmp1)
    tmp2 = sys.argv[len(sys.argv)-2].split(',')
    initial_point1.append(tmp2)
    tmp3 = sys.argv[len(sys.argv)-3].split(',')
    initial_point1.append(tmp3)
  
  initial_point = [list(map(int,i)) for i in initial_point1]
  
  if mode ==1:
    input = pd.read_csv(input_data, sep=",", header=None)
    data = {}
    keys =[]
    for row in input.itertuples(index=False):
        tmp = []
        for col in range(len(row)):
            if col == len(input.columns)-1:
                tmp = np.array(tmp)
                data[row[col]] =tmp
                keys.append(row[col])
            else:
                tmp.append(row[col])
    dist_res, inital_point = kmeans_algo(initial_point, data, dist_method, keys)
    for idx in dist_res:
      print("C"+str(idx+1),"=",dist_res[idx])
    print(inital_point)
    return

  if mode ==0:
    train_set = pd.read_csv(train_data, sep=",", header=None)
    test_set = pd.read_csv(test_data, sep=",", header=None)
    col_len = len(test_set.columns)
    total =[]
    for row in test_set.itertuples(index=False):
        point1 = []
        dist = []
        res = []
        for col in range(len(row)-1):
            point1.append(row[col])
            #print(row[col])
        for trow in train_set.itertuples(index=False):
            point2 = [] 
            #print(trow)
            for trcol in range(len(trow)-1):
                point2.append(trow[trcol])
            if dist_method==1:
                d1= euclidean_dist(np.array(point1),np.array(point2))
            else:
                d1=  manhattan(np.array(point1),np.array(point2))
            dist.append([d1, trow[len(trow)-1]])
        dist = sorted(dist,key=lambda x: (x[0]))
        #dist.pop(0)
        #print(dist)
        if voting ==1:
            tmp = unit_vote (dist, n_cluster)
        else:
            tmp = distance_vote(dist, n_cluster)
        total.append(tmp)
    test_set["predicted"] = total
  Label = set(list(train_set.iloc[:,col_len-1]))
  for idx in range(len(test_set)):
    print("want=",test_set.iloc[idx,col_len-1], "got=",test_set.iloc[idx,col_len] )
  for lbl in Label:
    real_label = test_set[test_set.iloc[:,col_len-1]==lbl].count()[0]
    predict_label =  test_set[(test_set['predicted']==lbl)].count()[0]
    correct_predict =  test_set[(test_set.iloc[:,col_len-1]==lbl) & (test_set['predicted']==lbl)].count()[0]
    print("Label=", lbl," Precision=",correct_predict,"/",predict_label  ," Recall=", correct_predict,"/",real_label)
  
  #print(initial_point)

  return
 
if __name__ == '__main__':
  main()

