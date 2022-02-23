kNN and kMeans

To execute this program:

To execute kNN: 
python3 lab4.py kNN [--k 3] [--dist e2] [--unitw] --train train.txt --test test.txt

- flags with '[ ]' is optinal flags! 
- Two hypens!
- To change manhattan distance : manh 
- flag '--dist' indicates whether choose euclidean (e2) or manhattan (manh)

To execute kmeans:
python3 lab4.py kMeans [--dist e2] --data data.txt 0,500 200,200 1000,1000
- 0,500 200,200 1000,1000 (x1, x2, x3) 
- between x1, x2, x3 points use space to seperate
- in a point, just use "comma" not space ex) (o) 0,500 (x) 0, 500 

Options for each flag:
--k : any positive integers
-- dist : 'e2' or 'manh'
--unitw : nothing is not needed (no option)
--data : file name for kmeans
--train : train dataset for knn
--test : test dataset for knn
