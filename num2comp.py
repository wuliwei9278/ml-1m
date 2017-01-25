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

def num2comp(filename, output, n_train, n_test):
  n_users = 0
  n_items = 0
  
  triples_list = []
  f = open(filename, 'r')
  for line in f:
    (user_id, item_id, rating) = line.strip().split(",")
    triples_list.append((int(user_id) + 1, int(item_id) + 1, float(rating)))
    n_users = max(n_users, int(user_id) + 1)
    n_items = max(n_items, int(item_id) + 1)
  f.close()

  print("Dataset for {0} users, {1} items loaded.".format(n_users, n_items)) 

  triples_list.sort(cmp=pair_comp)

  print("Dataset sorted.")

  idx = 0
  user_id = 0
  g1 = open(output + '_train.dat', 'w')
  # just to verify csv file is correct
  g4 = open(output + '_train_ratings.lsvm', 'w')

  g2 = open(output + '_train_ratings.csv', 'w')

  g3 = open(output + '_test_ratings.lsvm', 'w')  
  for u in xrange(1, n_users+1):
    ratings_list = []

    while triples_list[idx][0] == u:
      ratings_list.append((triples_list[idx][1], triples_list[idx][2]))
      idx = idx + 1
      if idx == len(triples_list):
        break



    if len(ratings_list) >= n_train + n_test:
      user_id = user_id + 1
      random.shuffle(ratings_list)
      train = ratings_list[:n_train]
      train.sort(cmp=pair_comp)
      test  = ratings_list[n_train:]
      test.sort(cmp=pair_comp)
      write_comps(g1, user_id, train)
      for i in xrange(len(train)):
        print(str(user_id) + "," + str(train[i][0]) + "," + str(int(train[i][1])), file = g2)
      write_lsvm(g4, user_id, train)
      write_lsvm(g3, user_id, test)

  g1.close()
  g2.close()
  g3.close()
  g4.close()
  
  print("Comparisons generated for {0} users {1} items".format(user_id, n_items))


if __name__ == "__main__":
  random.seed(1234)
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',
                      help="Dataset with user-item-rating triples")
  parser.add_argument('-o', '--output_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  parser.add_argument('-n', '--train_items', action='store', dest='n_train', type=int,
                      default=50, help="Number of training items per user (Default 50)") 
  parser.add_argument('-t', '--test_item', action='store', dest='n_test', type=int,
                      default=10, help="Minimum number of test items per user (Default 10)")
  parser.add_argument('-s', '--subsample', action='store_true',
                      help="At most (N_TRAIN) comparions from (N_TRAIN) ratings are sampled for each user")
  args = parser.parse_args()

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

  num2comp(args.input_file, args.output, args.n_train, args.n_test)
