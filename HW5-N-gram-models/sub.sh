ipython nbconvert --to=python language_model_improve.ipynb
# ipython nbconvert --to=python language_model_yezheng.ipynb
cp language_model_yezheng.py language_model_improve.py language_model_improve.ipynb hw5-writeup.pdf ../labels.txt sub.sh README.txt submission/ 
cd submission
zip ../submission.zip *