# python split.py MovieLens1m.csv -o ml1m
# python split.py netflix -o netflix
'''Generate training set and test set files (user, item, rating) triples'''
from __future__ import print_function
import os
import sys
import random
import argparse
import itertools

def pair_comp(x, y):
  if x[0] == y[0]:
    return x[1] - y[1]
  else:
    return x[0] - y[0]

def write_comps(f, user_id, ratings_list):
  for (rating1, rating2) in itertools.combinations(ratings_list, 2):
    if rating1[1] > rating2[1]:
      print(user_id, rating1[0], rating2[0], file=f)
    if rating1[1] < rating2[1]:
      print(user_id, rating2[0], rating1[0], file=f)

def write_lsvm(f, user_id, ratings_list):
  line = "" 
  for (item_id, rating) in ratings_list:
    line = line + "{0}:{1} ".format(item_id, rating)
  print(line, file=f)

def num2comp(filename, output):
  n_users = 0
  n_items = 0
  
  triples_list = []
  f = open(filename, 'r')
  for line in f:
    (user_id, item_id, rating) = line.strip().split()
    triples_list.append((int(user_id) + 1, int(item_id) + 1, float(rating)))
    n_users = max(n_users, int(user_id) + 1)
    n_items = max(n_items, int(item_id) + 1)
  f.close()

  print("Dataset for {0} users, {1} items loaded.".format(n_users, n_items)) 

  triples_list.sort(cmp=pair_comp)

  print("Dataset sorted.")

  idx = 0
  user_id = 0
  
  train100_dat = open(output + '_train100.dat', 'w')
  train200_dat = open(output + '_train200.dat', 'w')
  #train500_dat = open(output + '_train500.dat', 'w')
  #trainALL_dat = open(output + '_trainALL.dat', 'w')
  #train_dat = [train100_dat, train200_dat, train500_dat, trainALL_dat]

  train100_csv = open(output + '_train_ratings100.csv', 'w')
  train200_csv = open(output + '_train_ratings200.csv', 'w')
  #train500_csv = open(output + '_train_ratings500.csv', 'w')
  trainALL_csv = open(output + '_train_ratingsALL.csv', 'w')
  train_csv = [train200_csv, trainALL_csv]
  #train_csv = [train100_csv, train200_csv, train500_csv, trainALL_csv]
  test_lsvm = open(output + '_test_ratings.lsvm', 'w')  
  test_csv = open(output + '_test_ratings.csv', 'w')  
  for u in xrange(1, n_users+1):
    ratings_list = []

    while triples_list[idx][0] == u:
      ratings_list.append((triples_list[idx][1], triples_list[idx][2]))
      idx = idx + 1
      if idx == len(triples_list):
        break



    if len(ratings_list) >= 10 + 10:
      user_id = user_id + 1
      random.shuffle(ratings_list)
      test = ratings_list[:10]
      test.sort(cmp=pair_comp)
      
      train = ratings_list[10:]
      train.sort(cmp=pair_comp)
      for i in xrange(len(train)):
          print(str(user_id) + "," + str(train[i][0]) + "," + str(int(train[i][1])), file = trainALL_csv)
      train = train[:200]
      write_comps(train200_dat, user_id, train)
      for i in xrange(len(train)):
          print(str(user_id) + "," + str(train[i][0]) + "," + str(int(train[i][1])), file = train200_csv)
      train = train[:100]
      write_comps(train100_dat, user_id, train)
      for i in xrange(len(train)):
          print(str(user_id) + "," + str(train[i][0]) + "," + str(int(train[i][1])), file = train100_csv)

      write_lsvm(test_lsvm, user_id, test)
      for i in xrange(len(test)):
          print(str(user_id) + "," + str(test[i][0]) + "," + str(int(test[i][1])), file = test_csv)

  train100_dat.close()
  train200_dat.close()
  #train500_dat.close()
  #trainALL_dat.close()
  train100_csv.close()
  train200_csv.close()
  #train500_csv.close()
  trainALL_csv.close()
  test_lsvm.close()
  test_csv.close()
  
  
  print("Comparisons generated for {0} users {1} items".format(user_id, n_items))


if __name__ == "__main__":
  random.seed(1234)
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',
                      help="Dataset with user-item-rating triples")
  parser.add_argument('-o', '--output_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  
  args = parser.parse_args()

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

  num2comp(args.input_file, args.output)
