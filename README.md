### Large-scale Collaborative Ranking in Near-Linear Time

This repo consists of four folders:
- data folder: sample data “MovieLens1m.csv” can be found there, the data is of the form: “user id, movie id, rating”.
- util folder: python scripts to divide data into training and testing dataset (instructions on how to use these utilities functions are given below).
- code folder: Julia code which implements the primal-CR and primal-CR++ algorithm described in the paper is put into this folder (instructions on how to run the codes are given below).
- experimental folder: all the un-cleaned codes we initially wrote are put in this folder.




#### Instructions on how to run the code:
Our trained model can be tested in terms of NDCG@10, pairwise error and objective function.



1. Prepare a dataset of the form (user, item, ratings) triple in csv file. (Example: data/MovieLens1m.csv)

2. Run util/split1.py to get training data and test data by specifying the number of subsamples N to use as what we did in Section 5.1 in the paper 
**or** 
Run util/split2.py to get multiple training data of C = 100, 200, d2 but the same test data as what we did in Section 5.3:  

	
	```
	$ python util/split1.py data/MovieLens1m.csv -o ml1m -n 200
	```

	```
	$ python util/split2.py data/MovieLens1m.csv -o ml1m
	```

    , where the option -n specify the number of subsampled ratings per user N in Section 5.1 and -o specify the output file name prefix. The datasets generated will be in the current folder you type the command, i.e. the repo folder in the example. (The scripts also generate the training ratings which can be used for other methods)

3. Use command line to go to the repo folder and start Julia 

	```
	julia> julia -p 4
	```

	, where the option -p n provides n worker processes on the local machine. Use $ Julia -p 	1 for single thread experiments.


4. Type the following in Julia command line to load the functions for primal-CR algorithm:
	```
	julia> include("code/primalCR.jl")
	```
	Similarly, to run the primal-CR++ algorithm, type the following to include all necessary functions:
	```
	julia> include("code/primalCRpp.jl")
	```


5. Type in Julia command line 
	```
	julia> main("ml1m_train_ratings.csv", "ml1m_test_ratings.csv", 100, 5000)
	```
  	 Use `ctrl - c` to stop after it starts printing the results for the 1st iteration and 	type again 
	```
	julia> main("ml1m_train_ratings.csv", "ml1m_test_ratings.csv", 100, 5000)
	```
	One can replace the arguments for the main function by changing the training & test data 	file paths, and rank & lambda parameter in the model. The reason to type the same command 	twice in Julia command line is that the first time Julia will first compile the codes and 	the second time the codes will run much faster because the compilation time is saved.

