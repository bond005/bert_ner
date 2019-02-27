import os
import re
import sys
import unittest

import numpy as np

try:
    from bert_ner.bert_ner import BERT_NER
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from bert_ner.bert_ner import BERT_NER


class TestBertNer(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, 'ner'):
            del self.ner

    def test_creation(self):
        self.ner = BERT_NER()
        self.assertIsInstance(self.ner, BERT_NER)
        self.assertTrue(hasattr(self.ner, 'batch_size'))
        self.assertTrue(hasattr(self.ner, 'lr'))
        self.assertTrue(hasattr(self.ner, 'l2_reg'))
        self.assertTrue(hasattr(self.ner, 'bert_hub_module_handle'))
        self.assertTrue(hasattr(self.ner, 'finetune_bert'))
        self.assertTrue(hasattr(self.ner, 'max_epochs'))
        self.assertTrue(hasattr(self.ner, 'patience'))
        self.assertTrue(hasattr(self.ner, 'random_seed'))
        self.assertTrue(hasattr(self.ner, 'gpu_memory_frac'))
        self.assertTrue(hasattr(self.ner, 'max_seq_length'))
        self.assertTrue(hasattr(self.ner, 'validation_fraction'))
        self.assertTrue(hasattr(self.ner, 'verbose'))
        self.assertIsInstance(self.ner.batch_size, int)
        self.assertIsInstance(self.ner.lr, float)
        self.assertIsInstance(self.ner.l2_reg, float)
        self.assertIsInstance(self.ner.bert_hub_module_handle, str)
        self.assertIsInstance(self.ner.finetune_bert, bool)
        self.assertIsInstance(self.ner.max_epochs, int)
        self.assertIsInstance(self.ner.patience, int)
        self.assertIsNone(self.ner.random_seed)
        self.assertIsInstance(self.ner.gpu_memory_frac, float)
        self.assertIsInstance(self.ner.max_seq_length, int)
        self.assertIsInstance(self.ner.validation_fraction, float)
        self.assertIsInstance(self.ner.verbose, bool)

    def test_check_params_positive(self):
        BERT_NER.check_params(
            bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1', finetune_bert=True,
            batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10, patience=3,
            gpu_memory_frac=1.0, verbose=False, random_seed=42
        )
        self.assertTrue(True)

    def test_check_params_negative001(self):
        true_err_msg = re.escape('`bert_hub_module_handle` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                finetune_bert=True,
                batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10,
                patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative002(self):
        true_err_msg = re.escape('`bert_hub_module_handle` is wrong! Expected `{0}`, got `{1}`.'.format(
            type('abc'), type(123)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle=1, finetune_bert=True,
                batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1, max_epochs=10,
                patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative003(self):
        true_err_msg = re.escape('`batch_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative004(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size='32', max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative005(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=-3, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative006(self):
        true_err_msg = re.escape('`max_epochs` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative007(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs='10', patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative008(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=-3, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative009(self):
        true_err_msg = re.escape('`patience` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative010(self):
        true_err_msg = re.escape('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience='3', gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative011(self):
        true_err_msg = re.escape('`patience` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=-3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative012(self):
        true_err_msg = re.escape('`max_seq_length` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative013(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length='512', lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative014(self):
        true_err_msg = re.escape('`max_seq_length` is wrong! Expected a positive integer value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=-3, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative015(self):
        true_err_msg = re.escape('`validation_fraction` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative016(self):
        true_err_msg = re.escape('`validation_fraction` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction='0.1',
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative017(self):
        true_err_msg = '`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, but ' \
                       '{0} is not positive.'.format(-0.1)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=-0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative018(self):
        true_err_msg = '`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, but ' \
                       '{0} is not less than 1.0.'.format(1.1)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=1.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative019(self):
        true_err_msg = re.escape('`gpu_memory_frac` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, verbose=False, random_seed=42
            )

    def test_check_params_negative020(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac='1.0', verbose=False, random_seed=42
            )

    def test_check_params_negative021(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(-1.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=-1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative022(self):
        true_err_msg = re.escape('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                                 'but {0} is not proper.'.format(1.3))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.3, verbose=False, random_seed=42
            )

    def test_check_params_negative023(self):
        true_err_msg = re.escape('`lr` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative024(self):
        true_err_msg = re.escape('`lr` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr='1e-3', l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative025(self):
        true_err_msg = re.escape('`lr` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=0.0, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative026(self):
        true_err_msg = re.escape('`lr` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative027(self):
        true_err_msg = re.escape('`lr` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr='1e-3', l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative028(self):
        true_err_msg = re.escape('`lr` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=0.0, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative029(self):
        true_err_msg = re.escape('`l2_reg` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative030(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.5), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg='1e-4', validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative031(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected a non-negative floating-point value, but {0} is '
                                 'negative.'.format(-2.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=-2.0, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative032(self):
        true_err_msg = re.escape('`finetune_bert` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative033(self):
        true_err_msg = re.escape('`finetune_bert` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert='True', batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose=False, random_seed=42
            )

    def test_check_params_negative034(self):
        true_err_msg = re.escape('`verbose` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, random_seed=42
            )

    def test_check_params_negative035(self):
        true_err_msg = re.escape('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_params(
                bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                finetune_bert=True, batch_size=32, max_seq_length=512, lr=1e-3, l2_reg=1e-4, validation_fraction=0.1,
                max_epochs=10, patience=3, gpu_memory_frac=1.0, verbose='False', random_seed=42
            )

    def test_check_X_positive(self):
        X = ['abc', 'defgh', '4wdffg']
        BERT_NER.check_X(X, 'X_train')
        self.assertTrue(True)

    def test_check_X_negative01(self):
        X = {'abc', 'defgh', '4wdffg'}
        true_err_msg = re.escape('`X_train` is wrong, because it is not list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_X(X, 'X_train')

    def test_check_X_negative02(self):
        X = np.random.uniform(-1.0, 1.0, (10, 2))
        true_err_msg = re.escape('`X_train` is wrong, because it is not 1-D list!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_X(X, 'X_train')

    def test_check_X_negative03(self):
        X = ['abc', 23, '4wdffg']
        true_err_msg = re.escape('Item 1 of `X_train` is wrong, because it is not string-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_X(X, 'X_train')

    def text_check_Xy_positive(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_classes_list = ('LOC', 'ORG', 'PER')
        self.assertEqual(true_classes_list, BERT_NER.check_Xy(X, 'X_train', y, 'y_train'))

    def text_check_Xy_negative01(self):
        X = {
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        }
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('`X_train` is wrong, because it is not list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative02(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = {
            '1': {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            '2': {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        }
        true_err_msg = re.escape('`y_train` is wrong, because it is not a list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative03(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = np.random.uniform(-1.0, 1.0, (10, 2))
        true_err_msg = re.escape('`y_train` is wrong, because it is not 1-D list!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative04(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            },
            {
                'LOC': [(17, 24), (117, 130)]
            }
        ]
        true_err_msg = re.escape('Length of `X_train` does not correspond to length of `y_train`! 2 != 3')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative05(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            4
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because it is not a dictionary-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative06(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                1: [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because its key `1` is not a string-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative07(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'O': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because its key `O` incorrectly specifies a named '
                                 'entity!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative08(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                '123': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because its key `123` incorrectly specifies a named '
                                 'entity!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative09(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'loc': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because its key `loc` incorrectly specifies a named '
                                 'entity!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative10(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': {1, 2}
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because its value `{0}` is not a list-like '
                                 'object!'.format(y[0]['PER']))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative11(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), 63],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because named entity bounds `63` are not specified as '
                                 'list-like object!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative12(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77, 81)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 1 of `y_train` is wrong, because named entity bounds `{0}` are not specified as '
                                 '2-D list!'.format((63, 77, 81)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative13(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (219, 196)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because named entity bounds `{0}` are '
                                 'incorrect!'.format((219, 196)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative14(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(122, 137), (196, 519)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because named entity bounds `{0}` are '
                                 'incorrect!'.format((196, 519)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def text_check_Xy_negative15(self):
        X = [
            'Встреча с послом Италии в миде Грузии. По инициативе итальянской стороны чрезвычайный и полномочный посол '
            'Италии в Грузии Виторио Сандали встретился с заместителем министра иностранных дел Грузии Александром '
            'Налбандовым.',
            'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози. Как было объявлено, '
            'президент Франции прибыл в Вашингтон, чтобы обсудить с главой администрации США ряд насущных проблем, '
            'главное место среди которых занимает состояние мировой экономики и безопасность.'
        ]
        y = [
            {
                'ORG': [(26, 37)],
                'PER': [(-1, 137), (196, 219)]
            },
            {
                'ORG': [(126, 135)],
                'PER': [(0, 11), (63, 77)],
                'LOC': [(24, 34), (161, 178)]
            }
        ]
        true_err_msg = re.escape('Item 0 of `y_train` is wrong, because named entity bounds `{0}` are '
                                 'incorrect!'.format((-1, 137)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            BERT_NER.check_Xy(X, 'X_train', y, 'y_train')

    def test_calculate_bounds_of_tokens_positive01(self):
        source_text = 'Совершенно новую технологию перекачки российской водки за рубеж начали использовать ' \
                      'контрабандисты.'
        tokenized_text = ['Со', '##вер', '##шен', '##но', 'новую', 'тех', '##но', '##логи', '##ю', 'пер', '##ека',
                          '##чки', 'российской', 'вод', '##ки', 'за', 'р', '##уб', '##еж', 'начали', 'использовать',
                          'кон', '##тра', '##бан', '##ди', '##сты', '.']
        true_bounds = [(0, 2), (2, 5), (5, 8), (8, 10), (11, 16), (17, 20), (20, 22), (22, 26), (26, 27), (28, 31),
                       (31, 34), (34, 37), (38, 48), (49, 52), (52, 54), (55, 57), (58, 59), (59, 61), (61, 63),
                       (64, 70), (71, 83), (84, 87), (87, 90), (90, 93), (93, 95), (95, 98), (98, 99)]
        self.assertEqual(true_bounds, BERT_NER.calculate_bounds_of_tokens(source_text, tokenized_text))

    def test_calculate_bounds_of_tokens_positive02(self):
        source_text = 'Кстати за два дня до итальянцев, мальтийские пограничники уже задерживали лодку. ' \
                      'Однако они только дали им топливо, помогли завести двигатель и указали дорогу.'
        tokenized_text = ['К', '##стат', '##и', 'за', 'два', 'дня', 'до', 'итал', '##ья', '##нцев', ',', 'мал', '##ьт',
                          '##ий', '##ские', 'по', '##гра', '##ни', '##чники', 'уже', 'за', '##дер', '##живал', '##и',
                          'ло', '##дку', '.', 'Однако', 'они', 'только', 'дали', 'им', 'топ', '##ливо', ',', 'пом',
                          '##ог', '##ли', 'за', '##вести', 'д', '##вигатель', 'и', 'ук', '##аза', '##ли', 'дорог',
                          '##у', '.']
        true_bounds = [(0, 1), (1, 5), (5, 6), (7, 9), (10, 13), (14, 17), (18, 20), (21, 25), (25, 27), (27, 31),
                       (31, 32), (33, 36), (36, 38), (38, 40), (40, 44), (45, 47), (47, 50), (50, 52), (52, 57),
                       (58, 61), (62, 64), (64, 67), (67, 72), (72, 73), (74, 76), (76, 79), (79, 80), (81, 87),
                       (88, 91), (92, 98), (99, 103), (104, 106), (107, 110), (110, 114), (114, 115), (116, 119),
                       (119, 121), (121, 123), (124, 126), (126, 131), (132, 133), (133, 141), (142, 143), (144, 146),
                       (146, 149), (149, 151), (152, 157), (157, 158), (158, 159)]
        self.assertEqual(true_bounds, BERT_NER.calculate_bounds_of_tokens(source_text, tokenized_text))

    def test_calculate_bounds_of_tokens_positive03(self):
        source_text = 'Один из последних представителей клады, тираннозавр (Tyrannosaurus rex), живший 66–67 ' \
                      'миллионов лет назад, был одним из крупнейших когда-либо живших сухопутных хищников'
        tokenized_text = ['Один', 'из', 'последних', 'представителей', 'к', '##лады', ',', 'ти', '##ран', '##но',
                          '##за', '##вр', '(', 'Ty', '##ranno', '##saurus', 'rex', ')', ',', 'жив', '##ший', '66',
                          '[UNK]', '67', 'миллионов', 'лет', 'назад', ',', 'был', 'одним', 'из', 'крупнейших', 'когда',
                          '-', 'либо', 'жив', '##ших', 'су', '##хо', '##пу', '##тных', 'х', '##и', '##щ', '##ников']
        true_bounds = [(0, 4), (5, 7), (8, 17), (18, 32), (33, 34), (34, 38), (38, 39), (40, 42), (42, 45), (45, 47),
                       (47, 49), (49, 51), (52, 53), (53, 55), (55, 60), (60, 66), (67, 70), (70, 71), (71, 72),
                       (73, 76), (76, 79), (80, 82), (82, 83), (83, 85), (86, 95), (96, 99), (100, 105), (105, 106),
                       (107, 110), (111, 116), (117, 119), (120, 130), (131, 136), (136, 137), (137, 141), (142, 145),
                       (145, 148), (149, 151), (151, 153), (153, 155), (155, 159), (160, 161), (161, 162), (162, 163),
                       (163, 168)]
        self.assertEqual(true_bounds, BERT_NER.calculate_bounds_of_tokens(source_text, tokenized_text))

    def test_calculate_bounds_of_tokens_positive04(self):
        source_text = '–༼༽❆♖мама坦'
        tokenized_text = ['[UNK]', '[UNK]', '[UNK]', '[UNK]', '坦']
        true_bounds = [(0, 1), (1, 2), (2, 3), (3, 4), (9, 10)]
        self.assertEqual(true_bounds, BERT_NER.calculate_bounds_of_tokens(source_text, tokenized_text))

    def test_detect_token_labels_positive01(self):
        source_text = 'Барак Обама принимает в Белом доме своего французского коллегу Николя Саркози.'
        tokenized_text = ['Ба', '##рак', 'Об', '##ама', 'принимает', 'в', 'Б', '##елом', 'доме', 'своего',
                          'французского', 'кол', '##ле', '##гу', 'Н', '##ико', '##ля', 'Са', '##рко', '##зи', '.']
        token_bounds = BERT_NER.calculate_bounds_of_tokens(source_text, tokenized_text)
        indices_of_named_entities = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 1}
        y_true = np.array(
            [0, 2, 1, 1, 1, 0, 0, 4, 3, 3, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        y_pred = BERT_NER.detect_token_labels(token_bounds, indices_of_named_entities, label_IDs, 32)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_true.shape, y_pred.shape)
        self.assertEqual(y_true.tolist(), y_pred.tolist())

    def test_detect_token_labels_positive02(self):
        source_text = 'С 1876 г Павлов ассистирует профессору К. Н. Устимовичу в Медико-хирургической академии и ' \
                      'параллельно изучает физиологию кровообращения.'
        tokenized_text = ['С', '1876', 'г', 'Павло', '##в', 'а', '##сси', '##сти', '##рует', 'профессор', '##у', 'К',
                          '.', 'Н', '.', 'У', '##сти', '##мов', '##ич', '##у', 'в', 'М', '##еди', '##ко', '-',
                          'х', '##ир', '##ург', '##ической', 'академии', 'и', 'пара', '##лл', '##ельно',
                          'из', '##уч', '##ает', 'ф', '##из', '##ио', '##логи', '##ю',
                          'к', '##рово', '##об', '##ращения', '.']
        token_bounds = BERT_NER.calculate_bounds_of_tokens(source_text, tokenized_text)
        indices_of_named_entities = np.array(
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 3, 4: 2, 5: 4}
        y_true = np.array(
            [0, 0, 2, 1, 4, 3, 0, 0, 0, 0, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0, 8, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        y_pred = BERT_NER.detect_token_labels(token_bounds, indices_of_named_entities, label_IDs, 64)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_true.shape, y_pred.shape)
        self.assertEqual(y_true.tolist(), y_pred.tolist())

    def test_detect_token_labels_positive03(self):
        source_text = 'Весной 1890 года Варшавский и Томский университеты избирают его профессором.'
        tokenized_text = ['В', '##есной', '1890', 'года', 'В', '##ар', '##ша', '##вский', 'и', 'Томск', '##ий',
                          'университет', '##ы', 'из', '##бира', '##ют', 'его', 'профессором', '.']
        token_bounds = BERT_NER.calculate_bounds_of_tokens(source_text, tokenized_text)
        indices_of_named_entities = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0],
            dtype=np.int32
        )
        label_IDs = {1: 1, 2: 2, 3: 2}
        y_true = np.array(
            [0, 2, 1, 1, 1, 4, 3, 3, 3, 3, 4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int32
        )
        y_pred = BERT_NER.detect_token_labels(token_bounds, indices_of_named_entities, label_IDs, 32)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_true.shape, y_pred.shape)
        self.assertEqual(y_true.tolist(), y_pred.tolist())

    def test_tokenize_by_character_groups(self):
        source_text = 'Один из последних представителей клады, тираннозавр (Tyrannosaurus rex), живший 66–67 ' \
                      'миллионов лет назад, был одним из крупнейших когда-либо живших сухопутных хищников'
        true_tokens = ['Один', 'из', 'последних', 'представителей', 'клады', ',', 'тираннозавр', '(', 'Tyrannosaurus',
                       'rex', ')', ',', 'живший', '66', '–', '67', 'миллионов', 'лет', 'назад', ',', 'был', 'одним',
                       'из', 'крупнейших', 'когда', '-', 'либо', 'живших', 'сухопутных', 'хищников']
        self.assertEqual(true_tokens, BERT_NER.tokenize_by_character_groups(source_text))


if __name__ == '__main__':
    unittest.main(verbosity=2)
