# Copyright 2023 The myt5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add MyT5 Tasks to registry."""
import functools


# from multilingual_t5 import preprocessors
# from ultilingual_t5 import utils
# from multilingual_t5.evaluation import metrics as mt5_metrics
# from multilingual_t5.tasks import DEFAULT_OUTPUT_FEATURES as DEFAULT_MT5_OUTPUT_FEATURES

import seqio
import t5.data
import t5.data.tasks
import tensorflow_datasets as tfds
from t5x.myt5.vocabularies import MyteVocabulary
import t5x.myt5.preprocessors as preprocessors
from t5.evaluation import metrics
import numpy as np


MEAN_NOISE_SPAN_LENGTH = 20
DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

VOCABULARY = MyteVocabulary()
MASKING_PROBABILITY = 0.15

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(vocabulary=MyteVocabulary()),
    "targets": t5.data.Feature(vocabulary=MyteVocabulary())
}

FEATURE_MAP = {
    "myt5": DEFAULT_BYTE_OUTPUT_FEATURES,
}
# for testing
# MC4_LANGS = ["yi"]
MC4_LANGS = tfds.text.c4.MC4_LANGUAGES

MC4_VALIDATION_EXAMPLES = {
    'af': 1757, 'am': 289,'ar': 92374, 'az': 7206, 'be': 2103, 'bg': 32690, 'bg-Latn': 41, 'bn': 15130,
    'ca': 19562, 'ceb': 430, 'co': 211, 'cs': 82594, 'cy': 1016, 'da': 37071, 'de': 547566, 'el': 69435,
    'el-Latn': 171, 'en': 3933379, 'eo': 546, 'es': 592258, 'et': 10276, 'eu': 2077, 'fa': 81034, 'fi': 36512,
    'fil': 2381, 'fr': 453124, 'fy': 478, 'ga': 631, 'gd': 196, 'gl': 3811, 'gu': 1300, 'ha': 368, 'haw': 99,
    'hi': 26721, 'hi-Latn': 261, 'hmn': 175, 'ht': 246, 'hu': 56905, 'hy': 3804,'id': 19601, 'ig': 103, 'is': 3210,
    'it': 267322, 'iw': 17570, 'ja': 85618, 'ja-Latn': 221, 'jv': 253, 'ka': 3752, 'kk': 3443, 'km': 1359, 'kn': 1895,
    'ko': 24240, 'ku': 417, 'ky': 1188, 'la': 1630, 'lb': 856, 'lo': 290, 'lt': 18428, 'lv': 10034, 'mg': 254, 'mi': 156,
    'mk': 3713, 'ml': 3514, 'mn': 3021, 'mr': 4602, 'ms': 4719, 'mt': 1207, 'my': 1314, 'ne': 4738, 'nl': 137142,
    'no': 31134, 'ny': 121, 'pa': 692, 'pl': 178481, 'ps': 468, 'pt': 246120, 'ro': 66384, 'ru': 1014169, 'ru-Latn': 616,
    'sd': 206, 'si': 846, 'sk': 26882, 'sl': 12381, 'sm': 108, 'sn': 116, 'so': 1212, 'sq': 7057, 'sr': 4804, 'st': 103,
    'su': 151, 'sv': 63488, 'sw': 1296, 'ta': 5770, 'te': 2010, 'tg': 1526, 'th': 28062, 'tr': 133062, 'uk': 56321,
    'und': 3656588, 'ur': 3443, 'uz': 1259, 'vi': 132915, 'xh': 117, 'yi': 166, 'yo': 82, 'zh': 214733, 'zh-Latn': 492,
    'zu': 253}

# =========================== Pretraining Tasks/Mixtures =======================
# mC4

for lang in MC4_LANGS:
  seqio.TaskRegistry.add(
      "myt5_mc4.{}".format(lang.replace("-", "_")),
      source=seqio.TfdsDataSource(
          tfds_name="c4/multilingual:3.0.1",
          splits={
              "train": lang,
              "validation": f"{lang}-validation"
          }),
      preprocessors=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          functools.partial(t5.data.preprocessors.span_corruption,
                            mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[])

mc4 = ["myt5_mc4.{}".format(lang.replace("-", "_")) for lang in MC4_LANGS]
seqio.MixtureRegistry.add("myt5_mc4", mc4, default_rate=DEFAULT_MIX_RATE)

# =========================== LM EVALUATION =======================


def perplexities(targets, scores):
  """Computes perplexities for a batch of targets and scores."""
  bytes_in_targets = np.array([float(len(t.encode("utf-8"))) for t in targets])
  encoded_in_targets = np.array([float(len(VOCABULARY.encode(t))) for t in targets])

  return {
    "perplexity_bytes": seqio.metrics.Scalar(np.exp(-np.sum(scores) / np.sum(bytes_in_targets))),
    "perplexity_encoded": seqio.metrics.Scalar(np.exp(-np.sum(scores) / np.sum(encoded_in_targets))),
    "compression_factor": seqio.metrics.Scalar(np.sum(encoded_in_targets) / np.sum(bytes_in_targets))
  }


for lang in MC4_LANGS:
  seqio.TaskRegistry.add(
      "lm_mc4.{}".format(lang.replace("-", "_")),
      source=seqio.TfdsDataSource(
          tfds_name="c4/multilingual:3.0.1",
          splits={
              "train": lang,
              "validation": f"{lang}-validation[:{min(1000, MC4_VALIDATION_EXAMPLES[lang]//2)}]",
          }),
      preprocessors=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          t5.data.preprocessors.prefix_lm,
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[perplexities])

lm_mc4 = ["lm_mc4.{}".format(lang.replace("-", "_")) for lang in MC4_LANGS]
seqio.MixtureRegistry.add("lm_mc4", lm_mc4, 1.)

# # =========================== Fine-tuning Tasks/Mixtures =======================
# # ----- XNLI -----
# # XNLI zero-shot task. This fine-tunes on English MNLI training data and then
# # evaluates on multilingual XNLI dev/test data.

XNLI_LANGS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]

seqio.TaskRegistry.add(
    "myt5_xnli_train",
    source=seqio.TfdsDataSource(tfds_name="multi_nli:1.1.0", splits=["train"]),
    preprocessors=[
        preprocessors.process_mnli,
        *DEFAULT_PREPROCESSORS,
    ],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy])
for lang in XNLI_LANGS:
  seqio.TaskRegistry.add(
      "myt5_xnli_dev_test.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="xnli:1.1.0", splits=["validation", "test"]),
      preprocessors=[
          functools.partial(
              preprocessors.process_xnli, target_languages=[lang]),
          *DEFAULT_PREPROCESSORS,
      ],
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.accuracy])
  if lang == "en":
    continue
  seqio.TaskRegistry.add(
      "myt5_xnli_translate_train.{}".format(lang),
      source=seqio.TfdsDataSource(
          tfds_name="xtreme_xnli:1.1.0", splits=["train"]),
      preprocessors=[
          functools.partial(
              preprocessors.process_xnli, target_languages=[lang]),
          *DEFAULT_PREPROCESSORS,
      ],
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.accuracy])
seqio.TaskRegistry.add(
    "myt5_xnli_dev_test.all_langs",
    source=seqio.TfdsDataSource(
        tfds_name="xnli:1.1.0", splits=["validation", "test"]),
    preprocessors=[
        functools.partial(
            preprocessors.process_xnli, target_languages=XNLI_LANGS),
        *DEFAULT_PREPROCESSORS,
    ],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy])
xnli_zeroshot = (["myt5_xnli_train", "myt5_xnli_dev_test.all_langs"] +
                 ["myt5_xnli_dev_test.{}".format(lang) for lang in XNLI_LANGS])
seqio.MixtureRegistry.add("myt5_xnli_zeroshot", xnli_zeroshot, default_rate=1.0)
xnli_translate_train = xnli_zeroshot + [
    "myt5_xnli_translate_train.{}".format(lang)
    for lang in XNLI_LANGS
    if lang != "en"
]
seqio.MixtureRegistry.add(
    "myt5_xnli_translate_train", xnli_translate_train, default_rate=1.0)

# # ----- PAWS -----
# label_names = ["different_meaning", "paraphrase"]
# text_preprocessor = [
#     functools.partial(
#         t5.data.preprocessors.glue,
#         benchmark_name="paws",
#         label_names=label_names,
#         feature_names=["sentence1", "sentence2"],
#         id_key=None)
# ]

# postprocess_fn = functools.partial(
#         t5.data.postprocessors.string_label_to_class_id,
#         label_classes=label_names)

# seqio.TaskRegistry.add(
#     "myt5_paws",
#     source=seqio.TfdsDataSource(
#         tfds_name="paws_x_wiki/en:1.0.0", splits=["train"]),
#     preprocessors=text_preprocessor + DEFAULT_PREPROCESSORS,
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#     postprocess_fn=postprocess_fn,
#     metric_fns=[metrics.accuracy])

# for lang in utils.PAWSX_LANGS:
#   seqio.TaskRegistry.add(
#       "myt5_pawsx_dev_test.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="paws_x_wiki/{}:1.0.0".format(lang),
#           splits=["validation", "test"]),
#       preprocessors=text_preprocessor + DEFAULT_PREPROCESSORS,
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       postprocess_fn=postprocess_fn,
#       metric_fns=[metrics.accuracy])

#   # This uses machine translations provided by the PAWS-X paper.
#   seqio.TaskRegistry.add(
#       "myt5_pawsx_translate_train_original.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="paws_x_wiki/{}:1.0.0".format(lang), splits=["train"]),
#       preprocessors=text_preprocessor + DEFAULT_PREPROCESSORS,
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       postprocess_fn=postprocess_fn,
#       metric_fns=[metrics.accuracy])

#   if lang != "en":
#     # This uses machine translations provided by the XTREME paper.
#     seqio.TaskRegistry.add(
#         "myt5_pawsx_translate_train.{}".format(lang),
#         source=seqio.TfdsDataSource(
#             tfds_name="xtreme_pawsx/{}:1.0.0".format(lang), splits=["train"]),
#         preprocessors=text_preprocessor + DEFAULT_PREPROCESSORS,
#         output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#         postprocess_fn=postprocess_fn,
#         metric_fns=[metrics.accuracy])

# seqio.TaskRegistry.add(
#     "myt5_pawsx_dev_test.all_langs",
#     source=seqio.FunctionDataSource(
#         dataset_fn=utils.pawsx_all_langs_dataset_fn,
#         splits=["validation", "test"]),
#     preprocessors=text_preprocessor + DEFAULT_PREPROCESSORS,
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#     postprocess_fn=postprocess_fn,
#     metric_fns=[metrics.accuracy])

# # PAWSX Zero-Shot
# pawsx_eval = [
#     "myt5_pawsx_dev_test.{}".format(lang) for lang in utils.PAWSX_LANGS
# ] + ["myt5_pawsx_dev_test.all_langs"]
# pawsx = ["myt5_paws"] +  pawsx_eval
# seqio.MixtureRegistry.add("myt5_pawsx_zeroshot", pawsx, default_rate=1.0)

# pawsx_translate_train = ["myt5_paws"] + [
#     "myt5_pawsx_translate_train.{}".format(lang)
#     for lang in utils.PAWSX_LANGS
#     if lang != "en"
# ] + pawsx_eval
# seqio.MixtureRegistry.add(
#     "myt5_pawsx_translate_train", pawsx_translate_train, default_rate=1.0)

# pawsx_translate_train_original = [
#     "myt5_pawsx_translate_train_original.{}".format(lang)
#     for lang in utils.PAWSX_LANGS
# ] + pawsx_eval
# seqio.MixtureRegistry.add(
#     "myt5_pawsx_translate_train_original",
#     pawsx_translate_train,
#     default_rate=1.0)


# # ----- TyDiQA GoldP-----
# # The "validation" split contains all the validation examples for all the
# # individual languages together.
# TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
# seqio.TaskRegistry.add(
#     "myt5_tydiqa_train_dev",
#     source=seqio.TfdsDataSource(
#         tfds_name="tydi_qa/goldp:2.1.0", splits=["train", "validation"]),
#     preprocessors=[
#         preprocessors.xquad,
#         *DEFAULT_PREPROCESSORS,
#     ],
#     postprocess_fn=t5.data.postprocessors.qa,
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#     metric_fns=[metrics.squad])

# for lang in TYDIQA_LANGS:
#   seqio.TaskRegistry.add(
#       "myt5_tydiqa_dev.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="tydi_qa/goldp:2.1.0",
#           splits={"validation": "validation-{}".format(lang)}),
#       preprocessors=[
#           preprocessors.xquad,
#           *DEFAULT_PREPROCESSORS,
#       ],
#       postprocess_fn=t5.data.postprocessors.qa,
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       metric_fns=[metrics.squad])

# tydiqa = (["myt5_tydiqa_train_dev"] +
#           ["myt5_tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS])
# seqio.MixtureRegistry.add("myt5_tydiqa", tydiqa, default_rate=1.0)

# # ----- TyDiQA GoldP Zero-Shot-----
# # This Zero-Shot setting matches the XTREME setup, where training is done on
# # the English data of TyDiQa. In the TyDiQA paper, fine-tuning was done on
# # SQuAD for zero-shot evaluation.
# TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
# seqio.TaskRegistry.add(
#     "myt5_tydiqa_train.en",
#     source=seqio.TfdsDataSource(
#         tfds_name="tydi_qa/goldp:2.1.0", splits=["train"]),
#     preprocessors=[
#         preprocessors.xquad,
#         functools.partial(
#             preprocessors.filter_tydiqa_by_language, lang="english"),
#         *DEFAULT_PREPROCESSORS,
#     ],
#     postprocess_fn=t5.data.postprocessors.qa,
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#     metric_fns=[metrics.squad])


# tydiqa_zeroshot = (["myt5_tydiqa_train.en"] +
#                    ["myt5_tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS])
# seqio.MixtureRegistry.add(
#     "myt5_tydiqa_zeroshot", tydiqa_zeroshot, default_rate=1.0)


# # Defining translate-train tasks.
# for lang in TYDIQA_LANGS:
#   # Skipping English, since translate-train is not available.
#   if lang == "en":
#     continue
#   seqio.TaskRegistry.add(
#       "myt5_tydiqa_translate_train.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="tydi_qa/goldp:2.1.0",
#           splits={"train": "translate-train-{}".format(lang)}),
#       preprocessors=[
#           preprocessors.xquad,
#           *DEFAULT_PREPROCESSORS,
#       ],
#       postprocess_fn=t5.data.postprocessors.qa,
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       metric_fns=[metrics.squad])

# tydiqa_translate_train = (
#     ["myt5_tydiqa_train.en"]
#     + [f"myt5_tydiqa_translate_train.{lang}"
#        for lang in TYDIQA_LANGS if lang != "en"]
#     + [f"myt5_tydiqa_dev.{lang}" for lang in TYDIQA_LANGS])
# seqio.MixtureRegistry.add(
#     "myt5_tydiqa_translate_train", tydiqa_translate_train, default_rate=1.0)

# # ----- English SQUAD -----
# seqio.TaskRegistry.add(
#     "myt5_squad_train_dev",
#     source=seqio.TfdsDataSource(
#         tfds_name="squad/v1.1:3.0.0", splits=["train", "validation"]),
#     preprocessors=[
#         preprocessors.xquad,
#         *DEFAULT_PREPROCESSORS,
#     ],
#     postprocess_fn=t5.data.postprocessors.qa,
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#     metric_fns=[metrics.squad])

# # ----- XQuAD -----
# for lang in utils.XQUAD_LANGS_TRAIN_DEV:
#   seqio.TaskRegistry.add(
#       "myt5_xquad_translate_train_dev.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="xquad/{}:3.0.0".format(lang),
#           splits={
#               "train": "translate-train",
#               "validation": "translate-dev"
#           }),
#       preprocessors=[
#           preprocessors.xquad,
#           *DEFAULT_PREPROCESSORS,
#       ],
#       postprocess_fn=t5.data.postprocessors.qa,
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       metric_fns=[metrics.squad])

# for lang in utils.XQUAD_LANGS_TEST:
#   seqio.TaskRegistry.add(
#       "myt5_xquad_test.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="xquad/{}:3.0.0".format(lang), splits=["test"]),
#       preprocessors=[
#           preprocessors.xquad,
#           *DEFAULT_PREPROCESSORS,
#       ],
#       postprocess_fn=t5.data.postprocessors.qa,
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       metric_fns=[metrics.squad])

# # Additional test task containing all the languages.
# seqio.TaskRegistry.add(
#     "myt5_xquad_test.all_langs",
#     source=seqio.FunctionDataSource(
#         dataset_fn=utils.xquad_all_langs_dataset_fn, splits=["test"]),
#     preprocessors=[
#         preprocessors.xquad,
#         *DEFAULT_PREPROCESSORS,
#     ],
#     postprocess_fn=t5.data.postprocessors.qa,
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#     metric_fns=[metrics.squad])

# # XQuAD Zero-Shot (SQuAD train, SQuAD dev, XQuAD test).
# xquad_test = (["myt5_xquad_test.{}".format(lang)
#                for lang in utils.XQUAD_LANGS_TEST])
# xquad_zeroshot = (["myt5_squad_train_dev", "myt5_xquad_test.all_langs"] +
#                   xquad_test)
# seqio.MixtureRegistry.add(
#     "myt5_xquad_zeroshot", xquad_zeroshot, default_rate=1.0)

# # XQuAD Translate-Train (English SQuAD, XQuAD translate-train,
# # XQuAD translate-dev, XQuAD test)
# # Note that the QA translate-train baselines from Hu et al (XTREME)
# # do not include the English data. However, Fang et al (FILTER) do include
# # English data.
# xquad_translate_train = [
#     "myt5_xquad_translate_train_dev.{}".format(lang)
#     for lang in utils.XQUAD_LANGS_TRAIN_DEV
# ] + ["myt5_squad_train_dev"] +  ["myt5_xquad_test.all_langs"] + xquad_test
# seqio.MixtureRegistry.add(
#     "myt5_xquad_translate_train", xquad_translate_train, default_rate=1.0)

# # ----- MLQA -----
# MLQA_LANGS = ["ar", "de", "en", "es", "hi", "vi", "zh"]

# for lang in MLQA_LANGS:
#   seqio.TaskRegistry.add(
#       "myt5_mlqa_dev_test.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="mlqa/{}:1.0.0".format(lang), splits=["validation",
#                                                           "test"]),
#       preprocessors=[
#           preprocessors.xquad,
#           *DEFAULT_PREPROCESSORS,
#       ],
#       postprocess_fn=t5.data.postprocessors.qa,
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       metric_fns=[functools.partial(mt5_metrics.mlqa, lang=lang)])

# # MLQA Zero-Shot
# mlqa_dev_test = [f"myt5_mlqa_dev_test.{lang}" for lang in MLQA_LANGS]
# mlqa_zeroshot = ["myt5_squad_train_dev"] + mlqa_dev_test
# seqio.MixtureRegistry.add("myt5_mlqa_zeroshot", mlqa_zeroshot, default_rate=1.0)

# # MLQA Translate-Train
# mlqa_translate_train = [
#     "myt5_xquad_translate_train_dev.{}".format(lang)
#     for lang in MLQA_LANGS
#     if lang != "en"
# ] + ["myt5_squad_train_dev"] + mlqa_dev_test

# seqio.MixtureRegistry.add(
#     "myt5_mlqa_translate_train", mlqa_translate_train, default_rate=1.0)


# # Wikilingua
# # The version of Wikilingua here is the one from the GEM dataset:
# # https://arxiv.org/pdf/2102.01672.pdf.
# # Note: It was found that there is some data leakage in the GEM WikiLingua
# # data with English target summaries in the test set of one language
# # appearing in the training set of another language. Training in a multi-task
# # setup is not recommended till this issue is resolved.
# # TODO(adityabarua): Remove this comment when the issue is resolved.
# WIKILINGUA_LANGS = ["es", "ru", "tr", "vi"]

# for model, features in FEATURE_MAP.items():
#   for lang in WIKILINGUA_LANGS:
#     seqio.TaskRegistry.add(
#         "{model}_wikilingua.{lang}".format(model=model, lang=lang),
#         source=seqio.TfdsDataSource(
#             tfds_name="gem/wiki_lingua_{lang}_en:1.0.1".format(lang=lang)),
#         preprocessors=[
#             functools.partial(
#                 t5.data.preprocessors.rekey,
#                 key_map={"inputs": "source",
#                          "targets": "target"}),
#             *DEFAULT_PREPROCESSORS,
#         ],
#         output_features=features,
#         metric_fns=[metrics.rouge])

#   seqio.MixtureRegistry.add(
#       "{model}_wikilingua".format(model=model),
#       ["{model}_wikilingua.{lang}".format(model=model, lang=lang) for
#        lang in WIKILINGUA_LANGS],
#       default_rate=1.0)

# # SuperGLUE
# for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
#   # We use a simplified version of WSC, defined below
#   if "wsc" in b.name:
#     continue
#   if b.name == "axb":
#     text_preprocessor = [
#         functools.partial(
#             t5.data.preprocessors.rekey,
#             key_map={
#                 "premise": "sentence1",
#                 "hypothesis": "sentence2",
#                 "label": "label",
#                 "idx": "idx",
#             }),
#         t5.data.glue_utils.get_glue_text_preprocessor(b),
#     ]
#   else:
#     text_preprocessor = [t5.data.glue_utils.get_glue_text_preprocessor(b)]

#   seqio.TaskRegistry.add(
#       "myt5_super_glue_%s_v102" % b.name,
#       source=seqio.TfdsDataSource(
#           tfds_name="super_glue/%s:1.0.2" % b.name,
#           splits=["test"] if b.name in ["axb", "axg"] else None),
#       preprocessors=text_preprocessor + DEFAULT_PREPROCESSORS,
#       metric_fns=t5.data.glue_utils.get_super_glue_metric(b.name),
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       postprocess_fn=t5.data.glue_utils.get_glue_postprocess_fn(b))


# # Definitely Pronoun Resolution
# seqio.TaskRegistry.add(
#     "myt5_dpr_v001_simple",
#     source=seqio.TfdsDataSource(tfds_name="definite_pronoun_resolution:1.1.0"),
#     preprocessors=[
#         t5.data.preprocessors.definite_pronoun_resolution_simple,
#         *DEFAULT_PREPROCESSORS,
#     ],
#     metric_fns=[metrics.accuracy],
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# # WSC
# seqio.TaskRegistry.add(
#     "myt5_super_glue_wsc_v102_simple_train",
#     source=seqio.TfdsDataSource(
#         tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
#     preprocessors=[
#         functools.partial(
#             t5.data.preprocessors.wsc_simple, correct_referent_only=True),
#         *DEFAULT_PREPROCESSORS,
#     ],
#     metric_fns=[],
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# seqio.TaskRegistry.add(
#     "myt5_super_glue_wsc_v102_simple_eval",
#     source=seqio.TfdsDataSource(
#         tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation", "test"]),
#     preprocessors=[
#         functools.partial(
#             t5.data.preprocessors.wsc_simple, correct_referent_only=False),
#         *DEFAULT_PREPROCESSORS,
#     ],
#     postprocess_fn=t5.data.postprocessors.wsc_simple,
#     metric_fns=[metrics.accuracy],
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# _myt5_super_glue_tasks = {}
# for task, value in t5.data.get_super_glue_weight_mapping().items():
#   _myt5_super_glue_tasks["myt5_" + task] = value

# seqio.MixtureRegistry.add(
#     "myt5_super_glue_v102_proportional",
#     list(_myt5_super_glue_tasks.items()))


# # GLUE
# for b in tfds.text.glue.Glue.builder_configs.values():
#   seqio.TaskRegistry.add(
#       "myt5_glue_%s_v002" % b.name,
#       source=seqio.TfdsDataSource(
#           tfds_name="glue/%s:1.0.0" % b.name,
#           splits=["test"] if b.name == "ax" else None),
#       preprocessors=[
#           t5.data.glue_utils.get_glue_text_preprocessor(b),
#           *DEFAULT_PREPROCESSORS,
#       ],
#       metric_fns=t5.data.glue_utils.get_glue_metric(b.name),
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       postprocess_fn=t5.data.glue_utils.get_glue_postprocess_fn(b))

# seqio.MixtureRegistry.add(
#     "myt5_glue_v002_proportional",
#     list({
#         f"myt5_{k}": v
#         for k, v in t5.data.glue_utils.get_glue_weight_mapping().items()
#     }.items()))

# # ------ WMT ----------
# for prefix, b, tfds_version in t5.data.tasks.b_configs:
#   seqio.TaskRegistry.add(
#       "myt5_wmt%s_%s%s_v003" % (prefix, b.language_pair[1], b.language_pair[0]),
#       source=seqio.TfdsDataSource(tfds_name="wmt%s_translate/%s:%s" %
#                                   (prefix, b.name, tfds_version)),
#       preprocessors=[
#           functools.partial(
#               t5.data.preprocessors.translate,
#               source_language=b.language_pair[1],
#               target_language=b.language_pair[0],
#           ),
#           *DEFAULT_PREPROCESSORS,
#       ],
#       metric_fns=[metrics.bleu],
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# # Special case for t2t ende.
# b = tfds.translate.wmt_t2t.WmtT2tTranslate.builder_configs["de-en"]
# seqio.TaskRegistry.add(
#     "myt5_wmt_t2t_ende_v003",
#     source=seqio.TfdsDataSource(tfds_name="wmt_t2t_translate/de-en:1.0.0"),
#     preprocessors=[
#         functools.partial(
#             t5.data.preprocessors.translate,
#             source_language=b.language_pair[1],
#             target_language=b.language_pair[0],
#         ),
#         *DEFAULT_PREPROCESSORS,
#     ],
#     metric_fns=[metrics.bleu],
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# # ----- WikiAnn NER -----

# NER_LANGS = [
#     "af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi",
#     "fr", "he", "hi", "hu", "id", "it", "ja", "jv", "ka", "kk", "ko", "ml",
#     "mr", "ms", "my", "nl", "pt", "ru", "sw", "ta", "te", "th", "tl", "tr",
#     "ur", "vi", "yo", "zh"
# ]

# for lang in NER_LANGS:
#   seqio.TaskRegistry.add(
#       "myt5_ner_train.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="wikiann/{}:1.0.0".format(lang), splits=["train"]),
#       preprocessors=[
#           preprocessors.wikiann,
#           *DEFAULT_PREPROCESSORS,
#       ],
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       metric_fns=[mt5_metrics.span_f1])

#   seqio.TaskRegistry.add(
#       "myt5_ner_eval.{}".format(lang),
#       source=seqio.TfdsDataSource(
#           tfds_name="wikiann/{}:1.0.0".format(lang),
#           splits=["validation", "test"]),
#       preprocessors=[
#           preprocessors.wikiann,
#           *DEFAULT_PREPROCESSORS,
#       ],
#       output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
#       metric_fns=[mt5_metrics.span_f1])

# # NER zero-shot
# seqio.MixtureRegistry.add(
#     "myt5_ner_zeroshot", ["myt5_ner_train.{}".format("en")] +
#     ["myt5_ner_eval.{}".format(lang) for lang in NER_LANGS],
#     default_rate=1.0)

# # NER multilingual
# seqio.MixtureRegistry.add(
#     "myt5_ner_multilingual",
#     ["myt5_ner_train.{}".format(lang) for lang in NER_LANGS] +
#     ["myt5_ner_eval.{}".format(lang) for lang in NER_LANGS],
#     default_rate=1.0)

# # ----- GEM-XSum -----
# _rouge_fn = functools.partial(
#     metrics.rouge,
#     score_keys=["rouge1", "rouge2", "rougeL", "rougeLsum"])

# seqio.TaskRegistry.add(
#     "myt5_gem_xsum",
#     source=seqio.TfdsDataSource(tfds_name="gem/xsum:1.0.1"),
#     preprocessors=[
#         functools.partial(
#             t5.data.preprocessors.summarize,
#             article_key="document",
#             summary_key="target"),
#         *DEFAULT_PREPROCESSORS,
#     ],
#     metric_fns=[metrics.bleu, _rouge_fn],
#     output_features=DEFAULT_BYTE_OUTPUT_FEATURES)