# BERT-LSTM-SentimentAnalysis

Hugging face 'transformers' is required
```
pip install transformers
```

Then you can clone or download this code by 
```
%%shell
cd ./BERT-LSTM-SentimentAnalysis
chmod a+x download_dataset.sh
./download_dataset.sh
mv aclImdb ../
```

Run file 'train_py' with command 
```
python BERT-LSTM-SentimentAnalysis/train.py --freeze=true --nlayer=2 --data=imdb --learning_rate=0.0005
```
