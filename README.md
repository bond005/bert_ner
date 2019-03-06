# bert_ner
BERT-NER: named entity recognizer based on BERT and CRF.

The goal of this project is creation of a simple Python package with the sklearn-like interface for solution of different named entity recognition tasks in case number of labeled texts is very small (not greater than several thousands). Special neural network language model named as [BERT](https://arxiv.org/abs/1810.04805) (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) ensures this possibility, because this language model was pre-trained on large text corpora and so it can select deep semantic features from text.

## Getting Started

### Installing

To install this project on your local machine, you should run the following commands in Terminal:

```
git clone https://github.com/bond005/bert_ner.git
cd bert_ner
sudo python setup.py install
```

You can also run the tests

```
python setup.py test
```

### Usage

After installing the BERT-NER can be used as Python package in your projects. For example:

```
from bert_ner import BERT_NER  # import the BERT-NER package
ner = BERT_NER()  # create new named entity recognizer
```

For training of the BERT-NER you have to prepare list of short texts and do manual labeling of these texts by named entities. Result of text labeling should like as list of dictionaries, corresponding to list of source texts. Each dictionary in this list contains named entities and their bounds (initial and final character indices) in source text. For example:

```
texts_for_training = [
    'MIPT is one of the leading Russian universities in the areas of physics and technology.',
    'MIPT has a very rich history. Its founders included academicians Pyotr Kapitsa, Nikolay Semenov and Sergey Khristianovich.',
    'The main MIPT campus is located in Dolgoprudny, a northern suburb of Moscow.'
]
named_entity_labels_for_training = [
    {
        'ORG': [(0, 4)]
    },
    {
        'ORG': [(0, 4)],
        'PERSON': [(65, 78), (80, 95), (100, 121)]
    },
    {
        'LOCATION': [(35, 46), (69, 75)],
        'ORG': [(9, 13)]
    }
]
ner.fit(texts_for_training, named_entity_labels_for_training)
``` 

Predicted named entities for specified texts list also are presented in same format (as list of dictionaries):

```
texts_for_testing = [
    'Novosibirsk state university is located in the world-famous scientific center – Akademgorodok.',
    '"It’s like in the woods!" – that's what people say when they come to Akademgorodok for the first time.',
    'NSU’s new building will remind you of R2D2, the astromech droid and hero of the Star Wars saga'
]
results_of_prediction = ner.predict(texts_for_testing)
```

Quality evaluating of trained NER is based on examination all possible correspondences between a predicted labeling of named entities and the gold-standard labeling and choosing the best matches. After than special F1-score is calculated for all matched pairs "predicted entity" - "gold-standard entity" (fuzzy matching is taken into consideration too):

```
true_labels_for_testing = [
    {
        'LOCATION': [(80, 93)],
        'ORG': [(0, 28)]
    },
    {
        'LOCATION': [(69, 82)]
    },
    {
        'ORG': [(0, 3)],
        'PERSON': [(38, 42)]
    }
]
f1, precision, recall = ner.calculate_prediction_quality(true_labels_for_testing,
                                                         true_labels_for_testing,
                                                         ner.classes_list_)
``` 

You can serialize and de-serialize the BERT-NER object using the `pickle` module from Python’s standard library.

####Note

You have to use short texts such as sentences or small paragraphs, because long texts will be processed worse. If you train the BERT-NER on corpus of long texts, then the training can be converged slowly. If you use the BERT-NER, trained on short texts, for recognizing of long text, then only some initial words of this text can be tagged, and remaining words at the end of text will not be considered by algorithm. Besides, you need to use a very large volume of RAM for processing of long texts.

For solving of above-mentioned problem you can split long texts by shorter sentences using well-known NLP libraries such as [NLTK](http://www.nltk.org/api/nltk.tokenize.html?highlight=sent_tokenize#nltk.tokenize.sent_tokenize) or [SpaCy](https://spacy.io/api/token#is_sent_start).

##Demo

In the `demo` subdirectory you can see **demo_factrueval2016.py** - example of experiments on the FactRuEval-2016 text corpus, which is part of special competition devoted to named entity recognition and fact extraction in Russian (it is described in the paper [FactRuEval 2016: Evaluation of Named Entity Recognition and Fact Extraction Systems for Russian](http://www.dialog-21.ru/media/3430/starostinaetal.pdf)). In this example [multilingual BERT](https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1) is used as base neural model, and CRF is used as final classifier.

You can run this example on the command prompt in this way:

```
PYTHONPATH=$PWD python -u demo/demo_factrueval2016.py \
    -d /home/user/factRuEval-2016 \
    -m /home/user/FactRuEval2016_results/bert_and_crf.pkl \
    -r /home/user/FactRuEval2016_results/results_of_bert_and_crf \
    --max_epochs 1000 --batch 32
```

or in that way:

```
PYTHONPATH=$PWD python -u demo/demo_factrueval2016.py \
    -d /home/user/factRuEval-2016 \
    -m /home/user/FactRuEval2016_results/bert_and_crf.pkl \
    -r /home/user/FactRuEval2016_results/results_of_bert_and_crf \
    --max_epochs 1000 --batch 8 --finetune_bert
```

where:

- `/home/user/factRuEval-2016` is path to the FactRuEval-2016 repository cloned from https://github.com/dialogue-evaluation/factRuEval-2016;
- `/home/user/FactRuEval2016_results/tuned_bert_and_crf.pkl` is path to binary file into which the BERT-NER will be written after its training;
- `/home/user/FactRuEval2016_results/results_of_bert_and_crf` is path to directory with recognition results.

In first from above-mentioned ways you will train CRF only, and BERT will be 'frozen'. But in second way you will train both the BERT base and the CRF head. Second way is more hard and time-consuming, but it allows you to achieve better results. 

After recognition results calculation we can use the special FactRuEval-2016 script for evaluating of these results:

```
cd /home/user/factRuEval-2016/scripts
python t1_eval.py -t ~/FactRuEval2016_results/results_of_bert_and_crf -s ../testset/ -l
``` 

Quality score calculated by this script may differ from value returned by the `calculate_prediction_quality` method of the BERT-NER class.

## Acknowledgment

The work was supported by National Technology Initiative and PAO Sberbank project ID 0000000007417F630002.