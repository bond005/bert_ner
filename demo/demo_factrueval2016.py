from argparse import ArgumentParser
import codecs
import json
import logging
import os
import pickle
import sys
import tempfile

from typing import Union

try:
    from bert_ner.bert_ner import BERT_NER, bert_ner_logger
    from bert_ner.utils import factrueval2016_to_json, load_dataset
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from bert_ner.bert_ner import BERT_NER, bert_ner_logger
    from bert_ner.utils import factrueval2016_to_json, load_dataset


def train(factrueval2016_devset_dir: str, bert_will_be_tuned: bool, lstm_layer_size: Union[int, None], max_epochs: int,
          batch_size: int, gpu_memory_frac: float, model_name: str) -> BERT_NER:
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            recognizer = pickle.load(fp)
        assert isinstance(recognizer, BERT_NER)
        print('The NER has been successfully loaded from the file `{0}`...'.format(model_name))
        print('')
    else:
        temp_json_name = tempfile.NamedTemporaryFile(mode='w').name
        try:
            factrueval2016_to_json(factrueval2016_devset_dir, temp_json_name)
            X, y = load_dataset(temp_json_name)
        finally:
            if os.path.isfile(temp_json_name):
                os.remove(temp_json_name)
        print('Data for training have been loaded...')
        print('Number of samples is {0}.'.format(len(y)))
        print('')
        recognizer = BERT_NER(finetune_bert=bert_will_be_tuned, batch_size=batch_size, l2_reg=1e-4,
                              lstm_units=lstm_layer_size, validation_fraction=0.1, max_epochs=max_epochs, patience=3,
                              gpu_memory_frac=gpu_memory_frac, verbose=True, random_seed=42,
                              lr=1e-5 if bert_will_be_tuned else 1e-3)
        recognizer.fit(X, y)
        with open(model_name, 'wb') as fp:
            pickle.dump(recognizer, fp)
        print('')
        print('The NER has been successfully fitted and saved into the file `{0}`...'.format(model_name))
        print('')
    return recognizer


def recognize(factrueval2016_testset_dir: str, recognizer: BERT_NER, results_dir: str):
    temp_json_name = tempfile.NamedTemporaryFile(mode='w').name
    try:
        factrueval2016_to_json(factrueval2016_testset_dir, temp_json_name)
        with codecs.open(temp_json_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            data_for_testing = json.load(fp)
        _, true_entities = load_dataset(temp_json_name)
    finally:
        if os.path.isfile(temp_json_name):
            os.remove(temp_json_name)
    texts = []
    additional_info = []
    for cur_document in data_for_testing:
        base_name = os.path.join(results_dir, cur_document['base_name'] + '.task1')
        for cur_paragraph in cur_document['paragraph_bounds']:
            texts.append(cur_document['text'][cur_paragraph[0]:cur_paragraph[1]])
            additional_info.append((base_name, cur_paragraph))
    print('Data for final testing have been loaded...')
    print('Number of samples is {0}.'.format(len(true_entities)))
    print('')
    predicted_entities = recognizer.predict(texts)
    assert len(predicted_entities) == len(true_entities)
    f1, precision, recall = recognizer.calculate_prediction_quality(true_entities, predicted_entities,
                                                                    recognizer.classes_list_)
    print('F1-score is {0:.2%}.'.format(f1))
    print('Precision is {0:.2%}.'.format(precision))
    print('Recall is {0:.2%}.'.format(recall))
    results_for_factrueval_2016 = dict()
    for sample_idx, cur_result in enumerate(predicted_entities):
        base_name, paragraph_bounds = additional_info[sample_idx]
        for entity_type in cur_result:
            if entity_type == 'ORG':
                prepared_entity_type = 'org'
            elif entity_type == 'PERSON':
                prepared_entity_type = 'per'
            elif entity_type == 'LOCATION':
                prepared_entity_type = 'loc'
            else:
                prepared_entity_type = None
            if prepared_entity_type is None:
                raise ValueError('`{0}` is unknown entity type!'.format(entity_type))
            for entity_bounds in cur_result[entity_type]:
                postprocessed_entity = (
                    prepared_entity_type,
                    entity_bounds[0] + paragraph_bounds[0],
                    entity_bounds[1] - entity_bounds[0]
                )
                if base_name in results_for_factrueval_2016:
                    results_for_factrueval_2016[base_name].append(postprocessed_entity)
                else:
                    results_for_factrueval_2016[base_name] = [postprocessed_entity]
    for base_name in results_for_factrueval_2016:
        with codecs.open(base_name, mode='w', encoding='utf-8', errors='ignore') as fp:
            for cur_entity in sorted(results_for_factrueval_2016[base_name], key=lambda it: (it[1], it[2], it[0])):
                fp.write('{0} {1} {2}\n'.format(cur_entity[0], cur_entity[1], cur_entity[2]))


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the NER model.')
    parser.add_argument('-d', '--data', dest='data_name', type=str, required=True,
                        help='Path to the FactRuEval-2016 repository.')
    parser.add_argument('-r', '--result', dest='result_name', type=str, required=True,
                        help='The directory into which all recognized named entity labels will be saved.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, required=False, default=10,
                        help='Maximal number of training epochs.')
    parser.add_argument('--lstm', dest='lstm_units', type=int, required=False, default=None,
                        help='The LSTM layer size (if it is not specified, than the LSTM layer is not used).')
    parser.add_argument('--gpu_frac', dest='gpu_memory_frac', type=float, required=False, default=0.9,
                        help='Allocable part of the GPU memory for the NER model.')
    parser.add_argument('--finetune_bert', dest='finetune_bert', required=False, action='store_true',
                        default=False, help='Will be the BERT and CRF finetuned together? Or the BERT will be frozen?')
    args = parser.parse_args()

    devset_dir_name = os.path.join(os.path.normpath(args.data_name), 'devset')
    testset_dir_name = os.path.join(os.path.normpath(args.data_name), 'testset')
    recognizer = train(factrueval2016_devset_dir=devset_dir_name, bert_will_be_tuned=args.finetune_bert,
                       max_epochs=args.max_epochs, batch_size=args.batch_size, gpu_memory_frac=args.gpu_memory_frac,
                       model_name=os.path.normpath(args.model_name), lstm_layer_size=args.lstm_units)
    recognize(factrueval2016_testset_dir=testset_dir_name, recognizer=recognizer,
              results_dir=os.path.normpath(args.result_name))


if __name__ == '__main__':
    bert_ner_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    bert_ner_logger.addHandler(handler)
    main()
