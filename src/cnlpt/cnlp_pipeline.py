import os
import sys
import numpy as np
import shutil

from typing import Optional
from dataclasses import dataclass, field

from .cnlp_pipeline_utils import (
    model_dicts,
    get_sentences_and_labels,
    get_predictions,
)

from .cnlp_processors import classifier_to_relex, cnlp_compute_metrics, cnlp_processors

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from .formatting import tabulate_report

from tabulate import tabulate

from collections import defaultdict

from operator import itemgetter

from transformers import AutoConfig, AutoModel, HfArgumentParser

modes = ["inf", "eval"]


@dataclass
class PipelineArguments:
    """Pipeline data and model information and behavior"""

    models_dir: str = field(
        metadata={
            "help": (
                "Directory where each entity model and relation model is stored "
                "in a folder named after its "
                "corresponding shorthand name in cnlp_processors.py (e.g. rt_dose for RTDoseProcessor). "
                "NER models will be run first on the data after preprocessing, "
                "followed by relation models"
            )
        }
    )
    in_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to file:  \n\n In inference mode, file should contain"
                "one unlabeled sentence per line.\n\n"
                "In evaluation mode, file should contain one <label>\t<annotated sentence> "
                "per line."
            )
        },
    )
    in_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("Use this argument when running the pipeline over a folder with mutliple files.")},
    )
    out_dir: Optional[str] = field(
        default="cnlpt_predictions",
        metadata={"help": ("Output folder for predictions and metrics.")},
    )
    mode: str = field(
        default="inf",
        metadata={
            "help": (
                "Usage mode for the full pipeline:\n"
                "inference mode outputs annotated sentences "
                "and their relations. \n\n Eval mode"
                "outputs metrics for a provided set of samples "
                "(this mode requires labels)."
            )
        },
    )
    axis_task: str = field(
        default="rt_dose",
        metadata={
            "help": (
                "Shortened name of the task in cnlp_processors.py"
                "which generates the tag that will map to <main_mention_type-start> <mention> <main_mention_type-end>"
                "in pairwise annotations"
            )
        },
    )
    batch_size: int = field(
        default=1,
        metadata={"help": ("Batch size for pipeline batching (where batching helps, possibly for larger datasets)")},
    )


    

def labels_to_matrix(labels, dim):
    """
    List of tuples describing values at cells of a matrix 
    to the matrix they describe

    Args:
      labels: list of (idx_1, idx_2, label_idx) tuples
      dim: dimension for the resulting square matrix

    Returns:
      dim x dim matrix with, for each tuple, label_idx at cell idx_1, idx_2
    """
    source_matrix = np.array(labels)
    target_matrix = np.zeros((dim, dim), source_matrix.dtype)
    if not labels:
        return target_matrix
    target_matrix[source_matrix[:, 0], source_matrix[:, 1]] = source_matrix[:, 2]
    return target_matrix


def fix_gold_labels(labels, inv_map):
    """
    Rearranges the scoring matrix to prevent 
    any collisions of DOSE-BOOST and DOSE-SITE 
    (the only types of relations where this is possible)
    while retaining accurate scoring

    Args:
      labels: list of (idx_1, idx_2, label_idx)
      inv_map: dictionary for label_idx -> label

    Returns:
      labels where if a label is DOSE-SITE and shares indices with a DOSE-BOOST
      then its indices get reversed
    """

    if not labels:
        return labels

    boost_indices = set(
        [*map(lambda t: t[:2], filter(lambda s: inv_map[s[2]] == "DOSE-BOOST", labels))]
    )

    # if the DOSE-SITE shares indices  (i,j) with
    # a DOSE-BOOST send it to (j, i)
    # guaranteed to have no collisions since
    # all predictions by default are at some (i, j)
    # with i < j
    # so matrix diagonal and beneath is free real estate
    def adjust_indices(label_tuple):
        if not inv_map[label_tuple[2]] == "DOSE-SITE":
            return label_tuple
        first, second, label = label_tuple
        if (first, second) in boost_indices:
            return second, first, label
        return label_tuple

    adjusted_labels = [*map(adjust_indices, labels)]

    return adjusted_labels


def cnlpt_labels(paragraph_pairs):
    """
    Turns paragraph tuples into a scoring matrix

    Args:
      paragraph_pairs: list of (max paragraph length, list of labels in paragraph) by paragraph

    Returns:
      max paragraph length x number of paragraphs matrix
      where cells are populated from each paragraph's labels
    """

    def paragraph_matrix(pair):
        dim, labels = pair
        raw = labels_to_matrix(labels, dim)
        return np.ndarray.flatten(raw)

    return np.hstack([*map(paragraph_matrix, paragraph_pairs)])


def main():
    """Entry point"""
    parser = HfArgumentParser(PipelineArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script
        # and it's the path to a json file,
        # let's parse it to get our arguments.

        # the ',' is to deal with unary tuple weirdness
        (pipeline_args,) = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (pipeline_args,) = parser.parse_args_into_dataclasses()

    print("main entered")
        
    if pipeline_args.mode == "inf":
        inference(pipeline_args)
    elif pipeline_args.mode == "eval":
        evaluation(pipeline_args)
    else:
        ValueError("Invalid pipe mode!")


def inference(pipeline_args):
    """
    Inference mode entry point

    Args:
      pipeline_args: Pipeline information and behavior
    """
    # Required for loading cnlpt models using Huggingface Transformers
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    taggers_dict, out_model_dict = model_dicts(pipeline_args.models_dir)

    dir_mode = pipeline_args.in_dir is not None

    # https://stackoverflow.com/a/49237429
    # if there's a better way lmk
    if dir_mode:
        shutil.rmtree(pipeline_args.out_dir, ignore_errors=True)
        os.makedirs(pipeline_args.out_dir)

    file_list = (
        [
            os.path.join(pipeline_args.in_dir, in_fn)
            for in_fn in os.listdir(pipeline_args.in_dir)
        ]
        if dir_mode
        else [pipeline_args.in_file]
    )

    for in_file in file_list:
        # Only need raw sentences for inference
        _, sentences, _ = get_sentences_and_labels(
            in_file=in_file,
            mode="inf",
            task_names=out_model_dict.keys(),
        )
        paragraph_label_tuples = get_predictions(
            pipeline_args.out_dir,
            in_file,
            sentences,
            taggers_dict,
            out_model_dict,
            pipeline_args.axis_task,
        )


def evaluation(pipeline_args):
    print("entered eval")
    """

    Args:
      pipeline_args: data/model/behavior parameters

    Returns:
      None
    """
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    class_specific_keys = {"f1", "recall", "precision", "support"}

    taggers_dict, out_model_dict = model_dicts(pipeline_args.models_dir)

    print("models loaded")
    
    task_processor = cnlp_processors[classifier_to_relex[[*out_model_dict.keys()][0]]]()
    label_list = task_processor.get_labels()
    actual_labels = set()
    label_map = {label: i for i, label in enumerate(label_list)}
    inv_label_map = {i: label for i, label in enumerate(label_list)}
    dir_mode = pipeline_args.in_dir is not None

    if dir_mode:
        shutil.rmtree(pipeline_args.out_dir, ignore_errors=True)
        os.makedirs(pipeline_args.out_dir)

    document_cumulative_dict = defaultdict(lambda: defaultdict(lambda: []))

    file_list = (
        [
            os.path.join(pipeline_args.in_dir, in_fn)
            for in_fn in os.listdir(pipeline_args.in_dir)
        ]
        if dir_mode
        else [pipeline_args.in_file]
    )

    corpus_max_sent_len = -1

    ordered_metrics = {}

    total_labels = []
    total_preds = []

    def labels_str_2_int(labels):
        """

        Args:
          labels: list of (idx_1, idx_2, label_string)

        Returns:
           list of (idx_1, idx_2, label_idx)
        """

        def convert_label(label_tuple):
            """

            Args:
              label_tuple: (idx_1, idx_2, label_string)

            Returns:
              (idx_1, idx_2, label_idx)
            """
            first, second, label = label_tuple
            return first, second, label_map[label]

        return [*map(convert_label, labels)]

    # For eval need ground truth
    # labels as well as the length of
    # the longest sentence in the split
    # for matrix generation and padding
    for file_idx, in_file in enumerate(file_list):
        idx_labels_dict, sentences, split_max_len = get_sentences_and_labels(
            in_file=in_file,
            mode="eval",
            task_names=out_model_dict.keys(),
        )
        print(sentences)
        print(idx_labels_dict)

        pwd = os.getcwd()
        note_identifier = ".".join(os.path.basename(in_file).split(".")[:-1])
        note_dir = os.path.join(pwd, pipeline_args.out_dir + "/" + note_identifier)
        if not os.path.exists(note_dir):
            os.makedirs(note_dir)

        out_task, doc_labels_map = [*idx_labels_dict.items()][0]

        def fix_gold(label_list):
            """

            Args:
              label_list: list of label tuples

            Returns:
              list of labels with DOSE-BOOST <-> DOSE-SITE coordination
              to enable end2end scoring
            """
            return fix_gold_labels(label_list, inv_label_map)

        doc_labels = [*map(fix_gold, doc_labels_map)]

        present_rel_types = {
            inv_label_map[label]
            for label_ls in filter(None, doc_labels)
            for i_1, i_2, label in label_ls
        }.union({"None"})

        actual_labels.update(present_rel_types)

        print(f"\n\n{file_idx}.  {note_identifier}\n\n")

        def for_print(label_tuple):
            """

            Args:
              label_tuple: (idx_1, idx_2, label_idx)

            Returns:
              (idx_1, idx_2, label_string)
            """
            first, second, label_idx = label_tuple
            return f, s, inv_label_map[l]

        paragraph_label_tuples = get_predictions(
            pipeline_args.out_dir,
            in_file,
            sentences,
            taggers_dict,
            out_model_dict,
            pipeline_args.axis_task,
        )

        paragraph_lengths, label_lists = zip(*paragraph_label_tuples)

        pred_tuples = [*map(labels_str_2_int, label_lists)]
        gold_tuples = doc_labels

        def str_to_int(label_list):
            """

            Args:
              label_list: list of (idx_1, idx_2, label_idx)

            Returns:
              list of (idx_1, idx_2, label_string)
            """
            return [*map(for_print, label_list)]

        pred_pairs = zip(paragraph_lengths, pred_tuples)
        gold_pairs = zip(paragraph_lengths, gold_tuples)

        pred_matrix = cnlpt_labels(pred_pairs).astype("int")
        gold_matrix = cnlpt_labels(gold_pairs).astype("int")

        total_preds.append(pred_matrix)
        total_labels.append(gold_matrix)

        report = cnlp_compute_metrics(
            classifier_to_relex[out_task],
            pred_matrix,
            gold_matrix,
        )

        for score_type, score_list in report.items():
            if score_type in class_specific_keys:
                for score, category in zip(score_list, report["total_classes"]):
                    if category in present_rel_types:
                        document_cumulative_dict[score_type][category].append(score)

        report_str = tabulate_report(report)
        print(report_str)
        out_fn = "".join(
            [
                "metrics_" + note_identifier,
                ".txt",
            ]
        )

        out_file = os.path.join(note_dir, out_fn)
        ordered_metrics[file_idx] = note_identifier + "\n\n" + report_str
        with open(out_file, "wt") as out_writer:
            out_writer.write(report_str)

    if dir_mode:
        for out_task in out_model_dict:

            report = cnlp_compute_metrics(
                classifier_to_relex[out_task],
                np.hstack(total_preds),
                np.hstack(total_labels),
            )

            document_average_dict = defaultdict(lambda: defaultdict(lambda: 0))

            for (
                score_type,
                label_to_scores,
            ) in document_cumulative_dict.items():
                for label, scores in label_to_scores.items():
                    if len(scores) > 0:
                        if score_type != "support":
                            document_average_dict[score_type][label] = sum(
                                scores
                            ) / len(scores)
                        else:
                            document_average_dict["Number of Documents"][label] = len(
                                scores
                            )

            document_headers = ["Score", *sorted(actual_labels)]
            document_report_table = tabulate(
                [
                    [k, *itemgetter(*sorted(actual_labels))(v)]
                    for k, v in document_average_dict.items()
                ],
                headers=document_headers,
            )

            report_str = tabulate_report(report)
            print(f"FINAL -- SPLIT LEVEL INSTANCE AVERAGED RESULTS\n\n")

            print(report_str)
            print(f"\n\nFINAL -- SPLIT LEVEL DOCUMENT AVERAGED RESULTS\n\n")

            print(document_report_table)

            out_fn = "split_metrics.txt"

            out_file = os.path.join(pipeline_args.out_dir, out_fn)
            with open(out_file, "wt") as out_writer:
                out_writer.write("INSTANCE AVERAGED RESULTS OVER THE SPLIT\n\n")
                out_writer.write(report_str)

                out_writer.write("\n\DOCUMENT AVERAGED RESULTS OVER THE SPLIT\n\n")
                out_writer.write(document_report_table)

                out_writer.write(
                    "\n\n\nINDIVIDUAL DOCUMENT RESULTS (each available in identified folder)\n\n\n"
                )
                for doc_idx, name_and_scores in ordered_metrics.items():
                    out_writer.write(f"\n\n{doc_idx}.  {name_and_scores}\n\n")


if __name__ == "__main__":
    main()
