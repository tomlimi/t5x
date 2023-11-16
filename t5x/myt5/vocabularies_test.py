from absl.testing import absltest
from t5x.myt5 import vocabularies
import tensorflow as tf
import numpy as np


def _decode_tf(vocab, tokens):
  results = vocab.decode_tf(tf.constant(tokens, tf.int32)).numpy()

  def _apply(fun, sequence):
    if isinstance(sequence, (list, np.ndarray)):
      return [_apply(fun, x) for x in sequence]
    else:
      return fun(sequence)

  return _apply(lambda x: x.decode("UTF-8"), results)


class MyteVocabularyTest(absltest.TestCase):
  TEST_STRING = "this is a test 🤗 你好"
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

  def test_decode_tf(self):
    vocab = vocabularies.MyteVocabulary()

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
    vocab = vocabularies.MyteVocabulary()

    # Add two ids that are outside the allowed interval. They should be ignored.
    ids = tuple(list(self.TEST_BYTE_IDS) + [3000, -4000])
    expected_str = self.TEST_STRING

    self.assertEqual(expected_str, _decode_tf(vocab, ids))

  def test_decode_tf_invalid_byte_sequence(self):
    vocab = vocabularies.MyteVocabulary()

    # Add an invalid byte sequence, which should be ignored.
    ids = tuple(list(self.TEST_BYTE_IDS) + [0xC0, 0xC1])
    expected_str = self.TEST_STRING

    self.assertEqual(expected_str, _decode_tf(vocab, ids))

  def test_vocab(self):
    vocab = vocabularies.MyteVocabulary()
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
    vocab = vocabularies.MyteVocabulary()
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
