"""
Module containing processor classes, evaluation metrics, and output
modes for tasks defined in the library.

Add custom classes here to add new tasks to the library with the following steps:

#. Create a unique ``task_name`` for your task.
#. :data:`cnlp_output_modes` -- Add a mapping from a task name to a
   task type. Currently supported task types are sentence classification,
   tagging, relation extraction, and multi-task sentence classification.
#. Processor class -- Create a subclass of :class:`transformers.DataProcessor`
   for your data source. There are multiple examples to base off of,
   including intermediate abstractions like :class:`LabeledSentenceProcessor`,
   :class:`RelationProcessor`, :class:`SequenceProcessor`, that simplify
   the implementation.
#. :data:`cnlp_processors` -- Add a mapping from your task name to the
   "processor" class you created in the last step.
#. (Optional) -- Modify :func:`cnlp_compute_metrics` to add
   you task. If your task is classification a reasonable default will
   be used so this step would be optional.

.. data:: cnlp_processors

    Mapping from task names to processor classes

    :type: typing.Dict[str, transformers.DataProcessor]

.. data:: cnlp_output_modes

    Mapping from task names to output modes

    :type: typing.Dict[str, str]

"""
import os
import random
from os.path import basename, dirname
import time

# sklearn warnings are useless
def warn(*args, **kwargs):
    """

    Args:
      *args: 
      **kwargs: 

    Returns:

    """
    pass


import warnings

warnings.warn = warn


import sklearn
import logging
import json

from itertools import chain
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union, Any
from transformers.data.processors.utils import DataProcessor, InputExample
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.metrics import simple_accuracy
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score
import numpy as np
import numpy  # for Sphinx
from seqeval.metrics import (
    f1_score as seq_f1,
    classification_report as seq_cls,
    precision_score as seq_prec,
    recall_score as seq_rec,
)

logger = logging.getLogger(__name__)


def tagging_metrics(task_name, preds, labels):
    """One of the metrics functions for use in :func:`cnlp_compute_metrics`.
    
    Generates evaluation metrics for sequence tagging tasks.
    
    Ignores tags for which the true label is -100.
    
    The returned dict is structured as follows::
    
        {
            'acc': accuracy
            'token_f1': token-wise F1 score
            'f1': seqeval F1 score
            'report': seqeval classification report
        }

    Args:
      str: task_name: the task name used to index into cnlp_processors
      numpy: ndarray preds: the predicted labels from the model
      numpy: ndarray labels: the true labels

    Returns:
      a dictionary containing evaluation metrics

    """
    processor = cnlp_processors[task_name]()
    label_set = processor.get_labels()
    preds = np.array([*chain.from_iterable(preds)])#.flatten()
    labels = np.array([*chain.from_iterable(labels)])#.flatten().astype("int")
    # I don't even care anymore
    (pred_inds,) = np.where(labels != -100)

    preds = preds[pred_inds]
    labels = labels[pred_inds]

    pred_seq = [label_set[int(x)] for x in preds]
    label_seq = [label_set[int(x)] for x in labels]

    num_correct = (preds == labels).sum()

    acc = num_correct / len(preds)
    f1 = f1_score(labels, preds, average=None)

    return {
        "acc": acc,
        "token_f1": fix_np_types(f1),
        "f1": fix_np_types(seq_f1([label_seq], [pred_seq])),
        "precision": fix_np_types(seq_prec([label_seq], [pred_seq])),
        "recall": fix_np_types(seq_rec([label_seq], [pred_seq])),
        # etc
        # "doc_report": seq_prfs(
        #    [label_seq],
        #    [pred_seq],
        # ),
        "report": "\n" + seq_cls([label_seq], [pred_seq]),
    }


def relation_metrics(task_name, preds, labels):
    """One of the metrics functions for use in :func:`cnlp_compute_metrics`.
    
    Generates evaluation metrics for relation extraction tasks.
    
    Ignores tags for which the true label is -100.
    
    The returned dict is structured as follows::
    
        {
            'f1': F1 score
            'acc': accuracy
            'recall': recall
            'precision': precision
        }

    Args:
      str: task_name: the task name used to index into cnlp_processors
      numpy: ndarray preds: the predicted labels from the model
      numpy: ndarray labels: the true labels

    Returns:
      a dictionary containing evaluation metrics

    """

    processor = cnlp_processors[task_name]()
    label_set = processor.get_labels()

    # If we are using the attention-based relation extractor, many impossible pairs
    # are set to -100 so pytorch loss functions ignore them. We need to make sure the
    # scorer also ignores them.
    relevant_inds = np.where(labels != -100)
    relevant_labels = labels[relevant_inds].astype("int")
    relevant_preds = preds[relevant_inds].astype("int")

    ids_present_in_labels = fix_np_types(np.unique(relevant_labels))
    ids_present_in_predictions = fix_np_types(np.unique(relevant_preds))

    classes_present_in_labels = [label_set[int(i)] for i in ids_present_in_labels]
    classes_present_in_preds = [label_set[int(i)] for i in ids_present_in_predictions]

    total_classes = [
        label_set[i]
        for i in sorted({*ids_present_in_labels, *ids_present_in_predictions})
    ]

    num_correct = (relevant_labels == relevant_preds).sum()
    acc = num_correct / len(relevant_preds)

    recall = recall_score(y_pred=relevant_preds, y_true=relevant_labels, average=None)
    precision = precision_score(
        y_pred=relevant_preds, y_true=relevant_labels, average=None
    )
    f1_report = f1_score(y_true=relevant_labels, y_pred=relevant_preds, average=None)

    _, _, _, support = sklearn.metrics.precision_recall_fscore_support(
        y_true=relevant_labels, y_pred=relevant_preds, average=None
    )

    return {
        "f1": fix_np_types(f1_report),
        "acc": acc,
        "recall": fix_np_types(recall),
        "precision": fix_np_types(precision),
        "support": fix_np_types(support),
        "total_classes": total_classes,
        "classes present in labels": classes_present_in_labels,
        "classes present in predictions": classes_present_in_preds,
    }


def pipeline_relation_metrics(task_name, preds, labels):
    """One of the metrics functions for use in :func:`cnlp_compute_metrics`.
    
    Generates evaluation metrics for relation extraction tasks.
    
    Ignores tags for which the true label is -100.
    
    The returned dict is structured as follows::
    
        {
            'f1': F1 score
            'acc': accuracy
            'recall': recall
            'precision': precision
        }

    Args:
      str: task_name: the task name used to index into cnlp_processors
      numpy: ndarray preds: the predicted labels from the model
      numpy: ndarray labels: the true labels

    Returns:
      a dictionary containing evaluation metrics

    """

    processor = cnlp_processors[task_name]()
    label_set = processor.get_labels()

    ids_present_in_labels = fix_np_types(np.unique(labels))
    ids_present_in_predictions = fix_np_types(np.unique(preds))

    classes_present_in_labels = [label_set[i] for i in ids_present_in_labels]
    classes_present_in_preds = [label_set[i] for i in ids_present_in_predictions]

    total_classes = [
        label_set[i]
        for i in sorted({*ids_present_in_labels, *ids_present_in_predictions})
    ]

    num_correct = (labels == preds).sum()
    acc = num_correct / len(preds)

    recall = recall_score(y_pred=preds, y_true=labels, average=None)
    precision = precision_score(y_pred=preds, y_true=labels, average=None)
    f1_report = f1_score(y_true=labels, y_pred=preds, average=None)

    _, _, _, support = sklearn.metrics.precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average=None
    )

    return {
        "f1": fix_np_types(f1_report),
        "acc": acc,
        "recall": fix_np_types(recall),
        "precision": fix_np_types(precision),
        "support": fix_np_types(support),
        "total_classes": total_classes,
        "classes present in labels": classes_present_in_labels,
        "classes present in predictions": classes_present_in_preds,
    }


def fix_np_types(input_variable):
    """In the mtl classification setting, f1 is an array, and when the HF library
    tries to write out the training history to a json file it will throw an error.
    Here, we just check whether it's a numpy array and if so convert to a list.
    
    :meta private:

    Args:
      input_variable: 

    Returns:

    """
    if isinstance(input_variable, np.ndarray):
        return list(input_variable)

    return input_variable


def acc_and_f1(preds, labels):
    """One of the metrics functions for use in :func:`cnlp_compute_metrics`.
    
    Generates evaluation metrics for generic tasks.
    
    The returned dict is structured as follows::
    
        {
            'acc': accuracy
            'f1': F1 score
            'acc_and_f1': mean of accuracy and F1 score
            'recall': recall
            'precision': precision
        }

    Args:
      numpy: ndarray preds: the predicted labels from the model
      numpy: ndarray labels: the true labels

    Returns:
      a dictionary containing evaluation metrics

    """
    acc = simple_accuracy(np.array(preds), np.array(labels))
    recall = recall_score(y_true=labels, y_pred=preds, average=None)
    precision = precision_score(y_true=labels, y_pred=preds, average=None)
    f1 = f1_score(y_true=labels, y_pred=preds, average=None)

    # Need to generalize (or not)

    processor = cnlp_processors["rt_rel"]()

    label_set = processor.get_labels()

    ids_present_in_labels = fix_np_types(np.unique(labels))
    ids_present_in_predictions = fix_np_types(np.unique(preds))

    classes_present_in_labels = [label_set[int(i)] for i in ids_present_in_labels]
    classes_present_in_preds = [label_set[int(i)] for i in ids_present_in_predictions]

    total_classes = [
        label_set[int(i)]
        for i in sorted({*ids_present_in_labels, *ids_present_in_predictions})
    ]

    _, _, _, support = sklearn.metrics.precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average=None
    )

    return {
        "acc": fix_np_types(acc),
        "f1": fix_np_types(f1),
        "acc_and_f1": fix_np_types((acc + f1) / 2),
        "recall": fix_np_types(recall),
        "precision": fix_np_types(precision),
        "support": fix_np_types(support),
        "total_classes": total_classes,
        "classes present in labels": classes_present_in_labels,
        "classes present in predictions": classes_present_in_preds,
    }


tasks = {"polarity", "dtr", "alink", "alinkx", "tlink"}

dphe_tagging = {
    "dphe_med",
    "dphe_dosage",
    "dphe_duration",
    "dphe_form",
    "dphe_freq",
    "dphe_route",
    "dphe_strength",
}

rt_tagging = {
    "rt_boost",
    "rt_date",
    "rt_dose",
    "rt_fxfreq",
    "rt_fxno",
    "rt_site",
}


def cnlp_compute_metrics(task_name, preds, labels):
    """Function that defines and computes the metrics used for each task.
    
    When adding a task definition to this file, add a branch to this
    function defining what its evaluation metric invocation should be.
    If the new task is a simple classification task, a sensible default
    is defined; falling back on this will trigger a warning.

    Args:
      str: task_name: the task name used to index into cnlp_processors
      numpy: ndarray preds: the predicted labels from the model
      numpy: ndarray labels: the true labels

    Returns:
      a dictionary containing evaluation metrics

    """
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if (
        task_name == "polarity"
        or task_name == "uncertainty"
        or task_name == "history"
        or task_name == "subject"
    ):
        return acc_and_f1(preds, labels)
    elif task_name == "dtr":
        return acc_and_f1(preds, labels)
    elif task_name == "alink":
        return acc_and_f1(preds, labels)
    elif task_name == "alinkx" or task_name == "dphe_rel" or task_name == "rt_rel":
        return acc_and_f1(preds, labels)
    elif task_name == "tlink":
        return acc_and_f1(preds, labels)
    elif task_name == "conmod":
        return acc_and_f1(preds, labels)
    elif task_name == "timecat":
        return acc_and_f1(preds, labels)
    elif task_name.startswith("i2b22008"):
        return {
            "f1": fix_np_types(f1_score(y_true=labels, y_pred=preds, average=None))
        }  # acc_and_f1(preds, labels)
    elif (
        task_name == "timex"
        or task_name == "event"
        or task_name == "dphe"
        or task_name in dphe_tagging
        or task_name in rt_tagging
    ):
        return tagging_metrics(task_name, preds, labels)
    elif task_name == "tlink-sent" or task_name == "dphe_end2end":
        return relation_metrics(task_name, preds, labels)
    elif task_name == "rt_end2end":
        return pipeline_relation_metrics(task_name, preds, labels)
    elif cnlp_output_modes[task_name] == classification:
        logger.warn(
            "Choosing accuracy and f1 as default metrics; modify cnlp_compute_metrics() to customize for this task."
        )
        return acc_and_f1(preds, labels)
    else:
        raise Exception(
            "There is no metric defined for this task in function cnlp_compute_metrics()"
        )


class CnlpProcessor(DataProcessor):
    """Base class for single-task dataset processors

    Args:
      typing: Optional[typing.Dict[str, float]] downsampling: downsampling values for class balance

    Returns:

    """

    def __init__(self, downsampling=None):
        super().__init__()
        if downsampling is None:
            downsampling = {}
        self.downsampling = downsampling

    def get_one_score(self, results):
        """Return a single value to use as the score for
        selecting the best model epoch after training.

        Args:
          typing: Dict[str, typing.Any] results: the dictionary of evaluation
        metrics for the current epoch

        Returns:
          a single value; it needs to be of a type that can be
          ordered (preferably, but not necessarily, a float).

        """
        raise NotImplementedError()

    def get_example_from_tensor_dict(self, tensor_dict):
        """

        Args:
          tensor_dict: 

        Returns:

        """
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """

        Args:
          data_dir: 

        Returns:

        """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """

        Args:
          data_dir: 

        Returns:

        """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """

        Args:
          data_dir: 

        Returns:

        """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def _create_examples(self, lines, set_type, sequence=False, relations=False):
        """**This is an internal function, but it is included in the documentation
        to illustrate the input format for single-task datasets.**
        
        ----
        
        Creates examples for the training, dev and test sets from a
        headingless TSV file with one of the following structures:
        
        * For sequence classification::
        
            label\ttext
        
        * For sequence tagging::
        
            tag1 tag2 ... tagN\ttext
        
        * For relation tagging::
        
            <source1,target1> , <source2,target2> , ... , <sourceN,targetN>\ttext
        
        TODO: check that these formats are correct
        
        :meta public:

        Args:
          lines: 
          set_type: 
          sequence:  (Default value = False)
          relations:  (Default value = False)

        Returns:

        """
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if test_mode:
                # Some test sets have labels and some do not. discard the label if it has it but have to check so
                # we know which part of the line has the data.
                if line[0] in self.get_labels():
                    text_a = "\t".join(line[1:])
                else:
                    text_a = "\t".join(line[0:])
                label = None
            else:
                if sequence:
                    label = line[0].split(" ")
                elif relations:
                    if line[0].lower() == "none":
                        label = []
                    else:
                        label = [x[1:-1].split(",") for x in line[0].split(" , ")]
                else:
                    label = line[0]
                text_a = "\t".join(line[1:])

            if (
                set_type == "train"
                and not sequence
                and not relations
                and label in self.downsampling
            ):
                dart = random.random()
                # if downsampling is set to 0.1 then sample 10% of those instances.
                # so if our randomly generated number is bigger than our downsampling rate
                # we skip this instance.
                if dart > self.downsampling[label]:
                    continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples

    def get_num_tasks(self):
        """ """
        return 1


class LabeledSentenceProcessor(CnlpProcessor):
    """Base class for labeled sentence dataset processors"""

    def _create_examples(self, lines, set_type):
        """

        Args:
          lines: 
          set_type: 

        Returns:

        """
        return super()._create_examples(lines, set_type, sequence=False)

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"].mean()


class NegationProcessor(LabeledSentenceProcessor):
    """Processor for the negation datasets"""

    def get_labels(self):
        """ """
        return ["-1", "1"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"][1]


class UncertaintyProcessor(LabeledSentenceProcessor):
    """Processor for the negation datasets"""

    def get_labels(self):
        """ """
        return ["-1", "1"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"][1]


class HistoryProcessor(LabeledSentenceProcessor):
    """Processor for the negation datasets"""

    def get_labels(self):
        """ """
        return ["-1", "1"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"][1]


class DtrProcessor(LabeledSentenceProcessor):
    """Processor for DocTimeRel datasets"""

    def get_labels(self):
        """ """
        return ["BEFORE", "OVERLAP", "BEFORE/OVERLAP", "AFTER"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["acc"])


class AlinkxProcessor(LabeledSentenceProcessor):
    """Processor for an THYME ALINK dataset (links that describe change in temporal status of an event)
    The classifier version of the task is _given_ an event known to have some aspectual status, label that status.

    Args:

    Returns:

    """

    def get_labels(self):
        """ """
        return ["None", "CONTINUES", "INITIATES", "REINITIATES", "TERMINATES"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"][1:])


class AlinkProcessor(LabeledSentenceProcessor):
    """Processor for an THYME ALINK dataset (links that describe change in temporal status of an event)
    The classifier version of the task is _given_ an event known to have some aspectual status, label that status.

    Args:

    Returns:

    """

    def get_labels(self):
        """ """
        return ["CONTINUES", "INITIATES", "REINITIATES", "TERMINATES"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"])


class ContainsProcessor(LabeledSentenceProcessor):
    """Processor for narrative container relation (THYME). Describes the contains relation status between the
    two highlighted temporal entities (event or timex). NONE - no relation, CONTAINS - arg 1 contains arg2,
    CONTAINS-1 - arg 2 contains arg 1

    Args:

    Returns:

    """

    def get_labels(self):
        """ """
        return ["NONE", "CONTAINS", "CONTAINS-1"]


class TlinkProcessor(LabeledSentenceProcessor):
    """Processor for narrative container relation (THYME). Describes the contains relation status between the
    two highlighted temporal entities (event or timex). NONE - no relation, CONTAINS - arg 1 contains arg2,
    CONTAINS-1 - arg 2 contains arg 1

    Args:

    Returns:

    """

    def get_labels(self):
        """ """
        return ["BEFORE", "BEGINS-ON", "CONTAINS", "ENDS-ON", "OVERLAP"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"])


class DpheRelProcessor(LabeledSentenceProcessor):
    """ """
    def get_labels(self):
        """ """
        return [
            "None",
            "med-dosage",
            "med-duration",
            "med-form",
            "med-frequency",
            "med-route",
            "med-strength",
        ]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"][1:])


class RTRelProcessor(LabeledSentenceProcessor):
    """ """
    def get_labels(self):
        """ """
        return [
            "None",
            "DOSE-BOOST",
            "DOSE-DATE",
            "DOSE-DOSE",
            "DOSE-FXFREQ",
            "DOSE-FXNO",
            "DOSE-SITE",
        ]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"][1:])


class TimeCatProcessor(LabeledSentenceProcessor):
    """Processor for a THYME time expression dataset
    The classifier version of the task is _given_ a time class, label its time category (see labels below).

    Args:

    Returns:

    """

    def get_labels(self):
        """ """
        return [
            "DATE",
            "DOCTIME",
            "DURATION",
            "PREPOSTEXP",
            "QUANTIFIER",
            "SECTIONTIME",
            "SET",
            "TIME",
        ]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["acc"]


class ContextualModalityProcessor(LabeledSentenceProcessor):
    """Processor for a contextual modality dataset"""

    def get_labels(self):
        """ """
        return ["ACTUAL", "HYPOTHETICAL", "HEDGED", "GENERIC"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        # actual is the default and it's very common so we use the macro f1 of non-default categories for model selection.
        return np.mean(results["f1"][1:])


class UciDrugSentimentProcessor(LabeledSentenceProcessor):
    """Processor for the UCI Drug Review sentiment classification dataset"""

    def get_labels(self):
        """ """
        return ["Low", "Medium", "High"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"])


class Mimic_7_Processor(LabeledSentenceProcessor):
    """TODO: docstring"""

    def get_labels(self):
        """ """
        return ["7+", "7-"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"])


class Mimic_3_Processor(LabeledSentenceProcessor):
    """TODO: docstring"""

    def get_labels(self):
        """ """
        return ["3+", "3-"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"])


class CovidProcessor(LabeledSentenceProcessor):
    """TODO: docstring"""

    def get_labels(self):
        """ """
        return ["negative", "positive"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"][1]


class RelationProcessor(CnlpProcessor):
    """Base class for relation extraction dataset processors"""

    def _create_examples(self, lines, set_type):
        """

        Args:
          lines: 
          set_type: 

        Returns:

        """
        return super()._create_examples(lines, set_type, relations=True)


class DpheEndToEndProcessor(RelationProcessor):
    """ """
    def get_labels(self):
        """ """
        return [
            "None",
            "med-dosage",
            "med-duration",
            "med-form",
            "med-frequency",
            "med-route",
            "med-strength",
        ]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        # the 0th category is None
        return np.mean(results["f1"][1:])


class RTEndToEndProcessor(RelationProcessor):
    """ """
    def get_labels(self):
        """ """
        return [
            "None",
            "DOSE-BOOST",
            "DOSE-DATE",
            "DOSE-DOSE",
            "DOSE-FXFREQ",
            "DOSE-FXNO",
            "DOSE-SITE",
        ]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"][1:])


class TlinkRelationProcessor(RelationProcessor):
    """TODO: docstring"""

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        # the 0th category is None
        return np.mean(results["f1"][1:])

    def get_labels(self):
        """ """
        return ["None", "CONTAINS"]
        # return ['None', 'CONTAINS', 'NOTED-ON']
        # return ['None', 'CONTAINS', 'OVERLAP', 'BEFORE', 'BEGINS-ON', 'ENDS-ON']


class SequenceProcessor(CnlpProcessor):
    """Base class for sequence tagging dataset processors"""

    def _create_examples(self, lines, set_type):
        """

        Args:
          lines: 
          set_type: 

        Returns:

        """
        return super()._create_examples(lines, set_type, sequence=True)


class TimexProcessor(SequenceProcessor):
    """TODO: docstring"""

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return [
            "O",
            "B-DATE",
            "B-DURATION",
            "B-PREPOSTEXP",
            "B-QUANTIFIER",
            "B-SET",
            "B-TIME",
            "B-SECTIONTIME",
            "B-DOCTIME",
            "I-DATE",
            "I-DURATION",
            "I-PREPOSTEXP",
            "I-QUANTIFIER",
            "I-SET",
            "I-TIME",
            "I-SECTIONTIME",
            "I-DOCTIME",
        ]


class EventProcessor(SequenceProcessor):
    """TODO: docstring"""

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return [
            "O",
            "B-AFTER",
            "B-BEFORE",
            "B-BEFORE/OVERLAP",
            "B-OVERLAP",
            "I-AFTER",
            "I-BEFORE",
            "I-BEFORE/OVERLAP",
            "I-OVERLAP",
        ]
        # return ['B-EVENT', 'I-EVENT', 'O']


class DpheProcessor(SequenceProcessor):
    """TODO: docstring"""

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return [
            "O",
            "B-drug",
            "B-dosage",
            "B-duration",
            "B-frequency",
            "B-form",
            "B-route",
            "B-strength",
            "I-drug",
            "I-dosage",
            "I-duration",
            "I-frequency",
            "I-form",
            "I-route",
            "I-strength",
        ]


class DpheMedProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-medication", "I-medication"]


class DpheDosageProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-dosage", "I-dosage"]


class DpheDurationProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-duration", "I-duration"]


class DpheFormProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-form", "I-form"]


class DpheFrequencyProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-frequency", "I-frequency"]


class DpheRouteProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-route", "I-route"]


class DpheStrengthProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-strength", "I-strength"]


class RTProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return [
            "O",
            "B-Anatomical_site",
            "I-Anatomical_site",
            "B-RT_Fraction_Number",
            "I-RT_Fraction_Number",
            "B-FractionFrequency",
            "I-FractionFrequency",
            "B-RT_Dosage",
            "I-RT_Dosage",
            "B-RT_Date",
            "I-RT_Date",
        ]


class RTBoostProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-Boost", "I-Boost"]


class RTDateProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-RT_Date", "I-RT_Date"]


class RTDoseProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-RT_Dosage", "I-RT_Dosage"]


class RTFXFreqProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-FractionFrequency", "I-FractionFrequency"]


class RTFXNoProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-RT_Fraction_Number", "I-RT_Fraction_Number"]


class RTSiteProcessor(SequenceProcessor):
    """ """
    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return results["f1"]

    def get_labels(self):
        """ """
        return ["O", "B-Anatomical_site", "I-Anatomical_site"]


class MTLClassifierProcessor(DataProcessor):
    """Base class for multi-task learning classification dataset processors"""

    def get_classifiers(self):
        """Get the list of classification subtasks in this multi-task setting

        Args:

        Returns:
          a list of task names

        """
        return NotImplemented

    def get_num_tasks(self):
        """Get the number of subtasks in this multi-task setting.
        
        Equivalent to :obj:`len(self.get_classifiers())`.

        Args:

        Returns:
          the number of subtasks

        """
        return len(self.get_classifiers())

    def get_classifier_id(self):
        """Get the classifier ID name used in building the GUIDs for the
        :class:`transformers.InputExample` instances.
        
        Not necessarily equal to the ``task_name`` used as keys for
        :data:`cnlp_processors` and :data:`cnlp_output_modes`.

        Args:

        Returns:
          the value of the classifier ID

        """
        pass

    def get_default_label(self):
        """Get the default label to assign to unlabeled instances in the dataset.

        Args:

        Returns:
          the value of the default label

        """
        pass

    def get_example_from_tensor_dict(self, tensor_dict):
        """Not used.

        Args:
          tensor_dict: 

        Returns:

        """
        return RuntimeError("not implemented for MTL tasks")

    def get_train_examples(self, data_dir):
        """

        Args:
          data_dir: 

        Returns:

        """
        return self._get_json_examples(os.path.join(data_dir, "training.json"), "train")

    def get_dev_examples(self, data_dir):
        """

        Args:
          data_dir: 

        Returns:

        """
        return self._get_json_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_test_examples(self, data_dir):
        """

        Args:
          data_dir: 

        Returns:

        """
        return self._get_json_examples(os.path.join(data_dir, "test.json"), "test")

    def _get_json_examples(self, fn, set_type):
        """**This is an internal function, but it is included in the documentation
        to illustrate the input format for MTL datasets.**
        
        ----
        
        Creates examples for the training, dev and test sets
        from a JSON file with the following structure::
        
            {
                "<guid_1>": {
                    "text": "<text>",
                    "labels: {
                        "<task_1>": "<label>",
                        ...
                    }
                },
                ...
            }

        Args:
          str: fn: the path to the dataset file to load
          str: set_type: the type of split the file contains (e.g. train, dev, test)
          fn: 
          set_type: 

        Returns:
          the examples loaded from the file
          :meta public:

        """
        test_mode = set_type == "test"
        examples = []

        with open(fn, "rt") as f:
            data = json.load(f)

        for inst_id, instance in data.items():
            guid = "%s-%s" % (self.get_classifier_id(), inst_id)
            text_a = instance["text"]
            label_dict = instance["labels"]
            labels = [
                label_dict.get(x, self.get_default_label())
                for x in self.get_classifiers()
            ]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels)
            )

        return examples


class MimicRadiProcessor(MTLClassifierProcessor):
    """TODO: docstring"""

    def get_classifiers(self):
        """ """
        return ["3-", "3+", "7-", "7+"]
        # "3-": "Y", "3+": "N", "7-": "Y", "7+": "N"

    def get_labels(self):
        """ """
        # return [ ["Y", "N"] for x in range(len(self.get_classifiers()))]
        return ["Y", "N"]

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        print(results)
        # return results #['f1'].mean()
        return np.mean(results["f1"])

    def get_classifier_id(self):
        """ """
        return "mimic_radi"

    def get_default_label(self):
        """ """
        return "Unlabeled"


class i2b22008Processor(MTLClassifierProcessor):
    """Processor for the i2b2-2008 disease classification dataset"""

    def get_classifiers(self):
        """ """
        return [
            "Asthma",
            "CAD",
            "CHF",
            "Depression",
            "Diabetes",
            "Gallstones",
            "GERD",
            "Gout",
            "Hypertension",
            "Hypertriglyceridemia",
            "Hypercholesterolemia",
            "OA",
            "Obesity",
            "OSA",
            "PVD",
            "Venous Insufficiency",
        ]

    def get_labels(self):
        """ """
        # return [ ["Unlabeled", "Y", "N", "Q", "U"] for x in range(len(self.get_classifiers()))]
        return ["Unlabeled", "Y", "N", "Q", "U"]

    def get_default_label(self):
        """ """
        return "Unlabeled"

    def get_classifier_id(self):
        """ """
        return "i2b2-2008"

    def get_one_score(self, results):
        """

        Args:
          results: 

        Returns:

        """
        return np.mean(results["f1"])


"""
Add processor classes for new tasks here.
"""
cnlp_processors = {
    "polarity": NegationProcessor,
    "uncertainty": UncertaintyProcessor,
    "history": HistoryProcessor,
    "dtr": DtrProcessor,
    "alink": AlinkProcessor,
    "alinkx": AlinkxProcessor,
    "tlink": TlinkProcessor,
    "nc": ContainsProcessor,
    "timecat": TimeCatProcessor,
    "conmod": ContextualModalityProcessor,
    "timex": TimexProcessor,
    "event": EventProcessor,
    "tlink-sent": TlinkRelationProcessor,
    "i2b22008": i2b22008Processor,
    "ucidrug": UciDrugSentimentProcessor,
    "mimic_radi": MimicRadiProcessor,
    "mimic_3": Mimic_3_Processor,
    "mimic_7": Mimic_7_Processor,
    "covid": CovidProcessor,
    # DeepPhe medication and signatures tasks
    "dphe_end2end": DpheEndToEndProcessor,
    "dphe_rel": DpheRelProcessor,
    "dphe_multi": DpheProcessor,
    "dphe_med": DpheMedProcessor,
    "dphe_dosage": DpheDosageProcessor,
    "dphe_duration": DpheDurationProcessor,
    "dphe_form": DpheFormProcessor,
    "dphe_freq": DpheFrequencyProcessor,
    "dphe_route": DpheRouteProcessor,
    "dphe_strength": DpheStrengthProcessor,
    # Radiotherapy tasks
    "rt_end2end": RTEndToEndProcessor,
    "rt_rel": RTRelProcessor,
    "rt_multi": RTProcessor,
    "rt_boost": RTBoostProcessor,
    "rt_date": RTDateProcessor,
    "rt_dose": RTDoseProcessor,
    "rt_fxfreq": RTFXFreqProcessor,
    "rt_fxno": RTFXNoProcessor,
    "rt_site": RTSiteProcessor,
}


mtl = "mtl"
classification = "classification"
tagging = "tagging"
relex = "relations"


"""
Add output modes for new tasks here.
"""
cnlp_output_modes = {
    "polarity": classification,
    "uncertainty": classification,
    "history": classification,
    "dtr": classification,
    "alink": classification,
    "alinkx": classification,
    "tlink": classification,
    "nc": classification,
    "timecat": classification,
    "conmod": classification,
    "timex": tagging,
    "event": tagging,
    "tlink-sent": relex,
    "i2b22008": mtl,
    "ucidrug": classification,
    "mimic_radi": mtl,
    "mimic_3": classification,
    "mimic_7": classification,
    "covid": classification,
    # DeepPhe medication and signatures tasks
    "dphe_end2end": relex,
    "dphe_rel": classification,
    "dphe_multi": tagging,
    "dphe_med": tagging,
    "dphe_dosage": tagging,
    "dphe_duration": tagging,
    "dphe_form": tagging,
    "dphe_freq": tagging,
    "dphe_route": tagging,
    "dphe_strength": tagging,
    # Radiotherapy tasks
    "rt_end2end": relex,
    "rt_rel": classification,
    "rt_multi": tagging,
    "rt_boost": tagging,
    "rt_date": tagging,
    "rt_dose": tagging,
    "rt_fxfreq": tagging,
    "rt_fxno": tagging,
    "rt_site": tagging,
}

axis_tags = {
    # Medications and signatures
    "dphe_med": ("<a1>", "</a1>"),
    # Radiotherapy
    "rt_dose": ("<RT_DOSE-START>", "<RT_DOSE-END>"),
}

signature_tags = {
    # Medications and signatures
    "dphe_dosage": ("<a2>", "</a2>"),
    "dphe_duration": ("<a2>", "</a2>"),
    "dphe_form": ("<a2>", "</a2>"),
    "dphe_freq": ("<a2>", "</a2>"),
    "dphe_route": ("<a2>", "</a2>"),
    "dphe_strength": ("<a2>", "</a2>"),
    # Radiotherapy
    "rt_dose": ("<DOSE-START>", "<DOSE-END>"),  # Here for DOSE-DOSE
    "rt_boost": ("<BOOST-START>", "<BOOST-END>"),
    "rt_date": ("<DATE-START>", "<DATE-END>"),
    "rt_fxfreq": ("<FXFREQ-START>", "<FXFREQ-END>"),
    "rt_fxno": ("<FXNO-START>", "<FXNO-END>"),
    "rt_site": ("<SITE-START>", "<SITE-END>"),
}

# The class body for the relex label
# should be exactly the same as that of
# the classification label, the only difference besides
# their names is that the classification processor will inherit
# from LabeledSentenceProcessor and the relex will inherit
# from RelationProcessor
classifier_to_relex = {
    "dphe_rel": "dphe_end2end",
    "rt_rel": "rt_end2end",
}
