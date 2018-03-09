# key = '$1'
case "$1" in
0) 
    echo "simple VSM:"
    python evaluate.py data/dev_output.txt dev_output_features.txt
    ;;
1)
    echo "dense model:"
    python evaluate.py data/dev_output.txt dev_output_dense.txt
    ;;
#-----------------
#     echo "simple VSM:"
#     python evaluate.py data/dev_output.txt test_output_features.txt
#     echo "dense model:"
#     python evaluate.py data/dev_output.txt test_output_dense.txt
#     ;;
esac
# echo "leaderboard:"
# python evaluate.py data/dev_output.txt test_output_leaderboard.txt
#-------


ipython nbconvert --to=python sparse_vectorcluster.ipynb
ipython nbconvert --to=python dense_vectorcluster.ipynb

# cp question1.txt test_output_features.txt test_output_dense.txt Vocab.txt test_output_leaderboard.txt  /Users/yezheng/Documents/glove.6B/glove.6B.50d_yezheng.txt /Users/yezheng/Documents/glove.6B/glove.6B.100d_yezheng.txt /Users/yezheng/Documents/glove.6B/glove.6B.200d_yezheng.txt /Users/yezheng/Documents/glove.6B/glove.6B.300d_yezheng.txt /Users/yezheng/Documents/coocvec-500mostfreq-window-3-yezheng.vec /Users/yezheng/Documents/coocvec-500mostfreq-window-4-yezheng.vec /Users/yezheng/Documents/coocvec-500mostfreq-window-5-yezheng.vec /Users/yezheng/Documents/coocvec-600mostfreq-window-3-yezheng.vec /Users/yezheng/Documents/coocvec-1000mostfreq-window-3-yezheng.vec sparse_vectorcluster.py dense_vectorcluster.py sparse_vectorcluster.ipynb sparse_vectorcluster.ipynb writeup.pdf sub.sh submission/ 


cp question1.txt test_output_features.txt test_output_dense.txt Vocab.txt test_output_leaderboard.txt  /Users/yezheng/Documents/glove.6B/glove.6B.300d_yezheng.txt /Users/yezheng/Documents/coocvec-500mostfreq-window-4-yezheng.vec  sparse_vectorcluster.py dense_vectorcluster.py sparse_vectorcluster.ipynb sparse_vectorcluster.ipynb writeup.pdf sub.sh submission/ 
cd submission
zip ../submission.zip *