import tensorflow as tf


def xnli_map_hypothesis_premise(dataset, target_language):
  """Generates XNLI dataset with the hypothesis restricted to a target language.

  The XNLI dataset (https://www.tensorflow.org/datasets/catalog/xnli) contains
  the hypothesis in a TranslationVariableLanguages feature. The hypothesis looks
  like:
    hypothesis: {
      'language': ['ar', 'bg', 'en', ...],
      'translation': ['t1', 't2', 't3', ...]
    }
  This function processes this hypothesis to return a dataset of the form:
    {
      'language': 'ar',
      'translation': 't1',
      'label': '1'
    }
  The label is also extracted along with the hypothesis.

  Args:
    dataset: tf.data.Dataset to process.
    target_language: string, the target language to restrict the hypothesis in
      the dataset to.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    languages = x['hypothesis']['language']
    translations = x['hypothesis']['translation']

    # Create a tensor of the same length as languages so that everything can be
    # unbatched into examples for each language later.
    label = tf.fill(tf.shape(languages), x['label'])
    premise = tf.fill(tf.shape(languages), x['premise'][target_language])

    return {
        'language': languages,
        'translation': translations,
        'label': label,
        'premise': premise
    }

  dataset = dataset.map(
      _process, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
  dataset = dataset.filter(
      lambda x: tf.math.equal(x['language'], target_language))
  return dataset


def process_mnli(dataset):
  """Convert MNLI dataset into a text2text format.

  This function will return examples of the form:
  {
     'inputs': 'xnli: premise: <premise> hypothesis: <hypothesis>',
     'targets': '<target>'
  }

  Args:
    dataset: tf.data.Dataset to process.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    return {
        'inputs': tf.strings.join(['xnli: premise: ', x['premise'],
                                   ' hypothesis: ', x['hypothesis']]),
        'targets': tf.strings.as_string(x['label'])
    }

  return dataset.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def process_xnli(dataset, target_languages):
  """Processes the XNLI dataset into examples by language in a text2text format.

  The XNLI dataset contains examples of the form:
  {
        'hypothesis': {
            'language': ['lang1', 'lang2', 'lang3'],
            'translation': ['translation1', 'translation2', 'translation3'],
        },
        'label': 1,
        'premise': {
            'lang1': 'premise1',
            'lang2': 'premise2',
            'lang3': 'premise3'
        }
    }

  This function processes the XNLI dataset and returns examples of the form:
  {
    'inputs': 'xnli: premise: <premise> hypothesis: <hypothesis>',
    'targets': <target>
  }
  for each language in the list of input target languages.
  Args:
    dataset: tf.data.Dataset to process.
    target_languages: list of strings, the target languages.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    return {
        'inputs': tf.strings.join(['xnli: premise: ', x['premise'],
                                   ' hypothesis: ', x['translation']]),
        'targets': tf.strings.as_string(x['label'])
    }

  output = []
  for language in target_languages:
    examples = xnli_map_hypothesis_premise(dataset, target_language=language)
    d = examples.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    output.append(d)

  output_dataset = output[0]
  for lang_dataset in output[1:]:
    output_dataset = output_dataset.concatenate(lang_dataset)
  return output_dataset