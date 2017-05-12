This submission contains the following directories: 
* src/ - contains the python source files
* trace/ - contains output trace log for different runs under different sub-directories
* README.txt - instructions to run the program

*********** Running the program ************
****** Prerequisites ******
****** Installing required python packages ******

pip install pandas
pip install numpy
pip install scikit-learn
pip install nltk
pip install tqdm
pip install keras
pip install tensorflow
pip install pyemd
pip install fuzzywuzzy
pip install python-levenshtein
pip install --upgrade gensim
pip install pydot
pip install pydot2

python -m nltk.downloader stopwords
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader punkt

pip install h5py

****** Required data files ******
submit64.cs.utexas.edu:/scratch/cluster/pandian/data/ contains the following necessary preprocessing files required: 
1. glove.840B.300d.txt - glove embeddings
2. GoogleNews-vectors-negative300.bin.gz - word2vec on google news
3. quora_duplicate_questions.tsv - input file containing the training and test set
4. quora_features.csv - contains basic features like length of words, number of common words, fuzzy features etc.
5. quora_additional_features.csv - contains other features like POS tags, SRL tags, Chunk tags, verbs etc. 

**** Alternate way to download non-feature files ****
Files other than quora_features.csv and quora_additional_features.csv can also be downloaded online using the following steps: 

mkdir data
cd data
wget http://www-nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
wget http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

Note1: The preprocessed feature files can be obtained by running feature_engineering.py and additional_feature_engineering.py but it takes a lot of time to get the features. Hence it is advisable to use the extracted features file. 
Note2: The files in submit64.cs.utexas.edu:/scratch/cluster/pandian/data/ have global read permissions for all users and so using the path directory for --data should ideally work. If it doesn't work out, please copy the files to your home directory and specify it as the --data directory.

********** Command to run **********
cd src/
python -u deepnet.py [command_line_options] 

command_line_options:
--data=<data_path>	-	path containing the glove embeddings, questions and features files (default: ./data/)
--output=<output_path>	-	path to store the model checkpoint and the final plots (default: ./)
--baseline=<0|1> 	-	whether to run baseline model or not (default: 0)
--attention=<0|1>	-	whether to add attention model or not (default: 0)
--cnn=<0|1>		- 	whether to add CNN layer before LSTM or not (default: 0)
--regularization=<0|1>	-	whether to add regularization to LSTM or not (default: 0)
--bilstm=<0|1>		-	whether to use BiLSTM or single LSTM (default: 0)
--postags=<0|1>		-	whether to use POS tags model (default: 0)
--srltags=<0|1>		- 	whether to use SRL tags model (default: 0)
--epochs=<num_epochs>	-	number of epochs to train for (default: 2)
--chunk=<0|1>		-	whether to use chunk tags model  (default: 0)
--commonwords=<0|1>	-	whether to use number of common words model (default: 0)
--siamese=<0|1>		-	whether to use siamese network (default: 0)
--cwnolstm=<0|1>	-	whether to use LSTM for number of common words or just embedding layer (default: 0)

sample command: 
python -u deepnet.py --data=/scratch/cluster/pandian/data --output=run123  --epochs=10 --bilstm=1 --siamese=1

* the files present in the data directory will be used by the model for processing. 
* the checkpoint files and the accuracy, loss plots will be stored in the output directory.
