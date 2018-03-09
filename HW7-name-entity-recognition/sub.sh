rm submission.zip
rm submission/tagger.zip
ipython nbconvert --to=python ner-baseline.ipynb ner_ignacio.ipynb ner_crf.ipynb tagger/ner_LSTM.ipynb
cp ner_crf.ipynb ner_crf.py ner_ignacio.ipynb ner_ignacio.py sub.sh README.txt constrained_results.txt writeup.pdf ReADME.txt submission/ 
cp  tagger/ner_LSTM.ipynb tagger/ner_LSTM.py tagger/unconstrained_results.txt tagger/nn.py tagger/optimization.py tagger/train.py tagger/tagger.py tagger/utils.py tagger/loader.py   submission/tagger/
cp tagger/dataset/esp.testa tagger/dataset/esp.testb submission/tagger/dataset/
# cp -a /Users/yezheng/spanish.005/. submission/tagger/model/spanish.005
# cp -a tagger/evaluation/. submission/tagger/evaluation/
cd submission
cd tagger
zip -r ../tagger.zip *
cd ..
zip ../submission.zip *