ipython nbconvert --to=python language_model_NN_adaptive_learning_rate.ipynb language_model_NN_LSTM.ipynb GRU.ipynb
cp language_model_NN.py language_model_NN_adaptive_learning_rate.ipynb language_model_NN_LSTM.ipynb GRU.ipynb GRU.py hw6-writeup.pdf ../labels.txt sub.sh README.txt submission/ 
cd submission
zip ../submission.zip *