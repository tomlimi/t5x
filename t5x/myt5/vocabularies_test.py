from absl.testing import absltest
from t5x.myt5 import vocabularies
import tensorflow as tf
import numpy as np
import unicodedata
from tqdm import tqdm
import os


def _decode_tf(vocab, tokens):
  results = vocab.decode_tf(tf.constant(tokens, tf.int32)).numpy()

  def _apply(fun, sequence):
    if isinstance(sequence, (list, np.ndarray)):
      return [_apply(fun, x) for x in sequence]
    else:
      return fun(sequence)

  return _apply(lambda x: x.decode("UTF-8"), results)


class MyteVocabularyTest(absltest.TestCase):

  vocab = vocabularies.MyteVocabulary()
  TEST_STRING = "this is a test 🤗 你好"
  TEST_BYTE_IDS = (
      119,
      107,
      108,
      118,
      35,
      108,
      118,
      35,
      100,
      35,
      77,
      180,
      166,
      35,
      243,
      162,
      167,
      154,
      35,
      231,
      192,
      163,
      232,
      168,
      192,
  )

  TEST_FILE = os.path.join(os.path.dirname(__file__), "test_texts.txt")

  def test_decode_tf(self):
    vocab = self.vocab

    for rank in range(0, 3):
      ids = self.TEST_BYTE_IDS
      expected_str = self.TEST_STRING

      # Creates an arbitrarly nested tensor.
      for _ in range(rank):
        ids = [ids]
        expected_str = [expected_str]

      # single sequences to decode
      self.assertEqual(expected_str, _decode_tf(vocab, ids))

      # multiple sequences to decode
      res = _decode_tf(vocab, (ids, ids))
      exp = [expected_str] * 2
      self.assertEqual(exp, res)

  def test_decode_tf_oov_tokens(self):
    vocab = self.vocab

    # Add two ids that are outside the allowed interval. They should be ignored.
    ids = tuple(list(self.TEST_BYTE_IDS) + [3000, -4000])
    expected_str = self.TEST_STRING

    self.assertEqual(expected_str, _decode_tf(vocab, ids))

  def test_decode_tf_invalid_byte_sequence(self):
    vocab = self.vocab

    # Add an invalid byte sequence, which should be ignored.
    ids = tuple(list(self.TEST_BYTE_IDS) + [0xC0, 0xC1])
    expected_str = self.TEST_STRING

    self.assertEqual(expected_str, _decode_tf(vocab, ids))

  def test_vocab(self):
    vocab = self.vocab
    self.assertEqual(259, vocab.vocab_size)
    self.assertSequenceEqual(self.TEST_BYTE_IDS, vocab.encode(self.TEST_STRING))
    self.assertEqual(self.TEST_STRING, vocab.decode(self.TEST_BYTE_IDS))
    self.assertEqual(
        self.TEST_BYTE_IDS, tuple(vocab.encode_tf(self.TEST_STRING).numpy())
    )
    self.assertEqual(self.TEST_STRING, _decode_tf(vocab, self.TEST_BYTE_IDS))

  def test_extra_ids(self):
    vocab = vocabularies.MyteVocabulary(extra_ids=10)
    self.assertEqual(269, vocab.vocab_size)
    self.assertEqual("a", vocab.decode([100]))
    self.assertEqual("", vocab.decode([268]))

  def test_out_of_vocab(self):
    vocab = self.vocab
    self.assertEqual(259, vocab.vocab_size)
    self.assertEqual("", vocab.decode([260]))

  def test_equal(self):
    vocab1 = vocabularies.MyteVocabulary()
    vocab2 = vocabularies.MyteVocabulary()
    self.assertEqual(vocab1, vocab2)

  def test_not_equal(self):
    vocab1 = vocabularies.MyteVocabulary()
    vocab2 = vocabularies.MyteVocabulary(10)
    self.assertNotEqual(vocab1, vocab2)


class MyteVocabularyCorpusTest(absltest.TestCase):
    np.random.seed(42)
    vocab = vocabularies.MyteVocabulary()
    TEST_FILE = os.path.join(os.path.dirname(__file__), "test_texts.txt")
    BATCH_SIZE = 256

    def get_normalized_line_sample(cls, n_lines):
        with open(cls.TEST_FILE, "r") as corpus_file:
            lines = corpus_file.readlines()

        lines_normalized = [unicodedata.normalize('NFC', line.strip()) for line in lines]
        # sample
        return np.random.choice(lines_normalized, n_lines)

    def test_corpus(self):
        vocab = self.vocab

        normalized_lines = self.get_normalized_line_sample(100)

        for line in tqdm(normalized_lines):
            line.strip()
            encoded = vocab.encode_tf(tf.constant(line, dtype=tf.string))
            decoded_normalized = unicodedata.normalize('NFC',vocab.decode_tf(encoded).numpy().decode("UTF-8"))
            self.assertEqual(line, decoded_normalized)