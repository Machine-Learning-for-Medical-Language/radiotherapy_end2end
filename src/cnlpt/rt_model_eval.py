import os
import re
import torch
import warnings
import sys
import numpy as np
import shutil

from typing import Optional
from dataclasses import dataclass, field

from typing import Optional
from dataclasses import dataclass, field

from .cnlp_processors import (
    cnlp_processors,
    cnlp_output_modes,
    tagging,
    classification,
    classifier_to_relex,
    cnlp_compute_metrics,
    axis_tags,
    signature_tags,
)

from .pipelines.tagging import TaggingPipeline
from .pipelines.classification import ClassificationPipeline
from .pipelines import ctakes_tok

from .cnlp_pipeline_utils import (
    model_dicts,
    get_predictions,
)


from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from .formatting import tabulate_report

from tabulate import tabulate

from collections import defaultdict

from itertools import chain

from operator import itemgetter

from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser


SPECIAL_TOKENS = [
    "<e>",
    "</e>",
    "<a1>",
    "</a1>",
    "<a2>",
    "</a2>",
    "<RT_DOSE-START>",
    "<RT_DOSE-END>",
    "<DOSE-START>",
    "<DOSE-END>",
    "<DATE-START>",
    "<DATE-END>",
    "<BOOST-START>",
    "<BOOST-END>",
    "<FXFREQ-START>",
    "<FXFREQ-END>",
    "<FXNO-START>",
    "<FXNO-END>",
    "<SITE-START>",
    "<SITE-END>",
    "<cr>",
    "<neg>",
]

class_specific_keys = {"f1", "recall", "precision", "support"}


@dataclass
class RTEvalArguments:
    """ """

    model_dir: str = field(
        metadata={"help": ("Folder containing the trained model to be evaluated (with pytorch_model.bin etc).")}
    )
    in_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("Directory containing tsv files on which to evaluate the provided model.")},
    )
    # rt_rel 
    # rt_boost 
    # rt_date 
    # rt_dose 
    # rt_fxfreq 
    # rt_fxno 
    # rt_site
    task_name: str = field(
        metadata={
            "help": ("Shorthand task name from cnlp_processors.py describing format of the data on which to evaluate the model (see above comment in code for list)")
        },
    )
    batch_size: int = field(
        default=1,
        metadata={"help": ("Batch size for pipeline batching (where batching helps, possibly for larger datasets)")},
    )


def get_model(model_dir, task_name):
    """

    Args:
      model_dir: model location containing pytorch_model.bin
      task_name: cnlp_processors task name the model is being used for

    Returns:
      processing mode and model pipeline
    """

    if task_name not in cnlp_processors.keys():
        ValueError(f"Invalid task: {task_name}")
    main_device = 0 if torch.cuda.is_available() else -1

3    config = AutoConfig.from_pretrained(
        model_dir,
    )

    model = CnlpModelForClassification.from_pretrained(
        model_dir,
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        add_prefix_space=True,
        additional_special_tokens=SPECIAL_TOKENS,
    )

    task_processor = cnlp_processors[task_name]()

    if cnlp_output_modes[task_name] == tagging:
        return tagging, TaggingPipeline(
            model=model,
            tokenizer=tokenizer,
            task_processor=task_processor,
            device=main_device,
        )
    elif cnlp_output_modes[task_name] == classification:
        return classification, ClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            task_processor=task_processor,
            device=main_device,
        )
    else:
        ValueError((f"bad mode: {task_name}"))


def get_sentences_and_labels(in_file: str, task_name: str):
    """

    Args:
      str: in_file:  tsv file of label \t instance, separated by newlines
      str: task_name: cnlp processors task name for label generation schema

    Returns:
      list of sentences and a list of their corresponding labels
    """
    # 'dev' lets us get labels without running into issues of downsampling

    task_processor = cnlp_processors[task_name]()

    lines = task_processor._read_tsv(in_file)
    examples = task_processor._create_examples(lines, "dev")

    if len(examples) == 0:
        return [], []

    label_map = {label: i for i, label in enumerate(task_processor.get_labels())}

    if examples[0].text_b is None:
        sentences = [example.text_a for example in examples]
    else:
        sentences = [(example.text_a, example.text_b) for example in examples]

    if cnlp_output_modes[task_name] == tagging:
        labels = [[label_map[t_l] for t_l in example.label] for example in examples]
    elif cnlp_output_modes[task_name] == classification:
        labels = [label_map[example.label] for example in examples]
    else:
        labels = None
        ValueError((f"bad mode: {task_name}"))
    return sentences, labels


def main():
    """Entry point"""
    parser = HfArgumentParser(RTEvalArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (rt_eval_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (rt_eval_args,) = parser.parse_args_into_dataclasses()

    model_eval(eval_args)


def model_eval(eval_args):
    """

    Args:
      eval_args: model, data, and behavior arguments

    Returns:

    """
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    mode, pipeline = get_model(eval_args.model_dir, eval_args.task_name)

    re_document_cumulative_dict = defaultdict(lambda: defaultdict(lambda: []))
    ner_document_cumulative_dict = defaultdict(lambda: [])

    task_processor = cnlp_processors[eval_args.task_name]()
    label_map = {label: i for i, label in enumerate(task_processor.get_labels())}
    file_list = [
        os.path.join(eval_args.in_dir, in_fn) for in_fn in os.listdir(eval_args.in_dir)
    ]

    def note_id(fn):
        """

        Args:
          fn: actual filename

        Returns:
          base filename without extension
        """
        return ".".join(os.path.basename(fn).split(".")[:-1])

    def rank(fn):
        """

        Args:
          fn: actual filename

        Returns:
          component of the file by which we want to sort the files
        """
        real_name = note_id(fn)
        if real_name[:4].lower() == "test" and real_name.lower() != "test":
            return int(real_name[4:])
        return real_name

    for file_idx, in_file in enumerate(sorted(file_list, key=lambda s: rank(s))):
        sentences, labels = get_sentences_and_labels(
            in_file,
            eval_args.task_name,
        )

        labels_in_doc = (
            sorted(set(chain.from_iterable(labels)))
            if mode == tagging
            else sorted(set(labels))
        )

        note_identifier = note_id(in_file)

        print(f"\n\n{file_idx}.  {note_identifier}\n\n")

        if any(labels_in_doc):

            if mode == classification:
                predictions = pipeline(
                    sentences,
                    padding="max_length",
                    truncation=True,
                    is_split_into_words=True,
                    batch_size=eval_args.batch_size,
                )
                report = cnlp_compute_metrics(
                    eval_args.task_name,
                    [
                        label_map[p]
                        for p in [
                            max(d, key=lambda s: s["score"])["label"]
                            for d in chain.from_iterable(predictions)
                        ]
                    ],
                    labels,
                )

                for score_type, score_list in report.items():
                    if score_type in class_specific_keys:
                        for score, category in zip(score_list, report["total_classes"]):
                            if category in report["classes present in labels"]:
                                re_document_cumulative_dict[score_type][
                                    category
                                ].append(score)

                report_str = tabulate_report(report)
                print(report_str)
            elif mode == tagging:
                predictions = pipeline(
                    sentences,
                    padding="max_length",
                    truncation=True,
                    # is_split_into_words=True,
                    batch_size=eval_args.batch_size,
                )

                report = cnlp_compute_metrics(
                    eval_args.task_name,
                    [
                        [label_map[l] for l in p]
                        for p in chain.from_iterable(predictions)
                    ],
                    labels,
                )

                print(report["report"])

                print(
                    f"Stored scores: precicion \t {report['precision']} recall \t {report['recall']} f1 \t {report['f1']}"
                )
                ner_document_cumulative_dict["precision"].append(report["precision"])
                ner_document_cumulative_dict["recall"].append(report["recall"])
                ner_document_cumulative_dict["f1"].append(report["f1"])

    if mode == classification:
        document_average_dict = defaultdict(lambda: defaultdict(lambda: 0))

        for (
            score_type,
            label_to_scores,
        ) in re_document_cumulative_dict.items():
            for label, scores in label_to_scores.items():
                if len(scores) > 0:
                    if score_type != "support":
                        document_average_dict[score_type][label] = sum(scores) / len(
                            scores
                        )
                    else:
                        document_average_dict["Number of Documents"][label] = len(
                            scores
                        )

        document_headers = ["Score", *task_processor.get_labels()]
        document_report_table = tabulate(
            [
                [k, *itemgetter(*task_processor.get_labels())(v)]
                for k, v in document_average_dict.items()
            ],
            headers=document_headers,
        )
        print(f"\n\nFINAL -- SPLIT LEVEL DOCUMENT AVERAGED RESULTS\n\n")

        print(document_report_table)
    elif mode == tagging:
        document_average_dict = defaultdict(lambda: defaultdict(lambda: 0))
        num_docs = -1
        for (
            score_type,
            scores,
        ) in ner_document_cumulative_dict.items():
            if len(scores) > 0:
                document_average_dict[score_type] = sum(scores) / len(scores)
                num_docs = len(scores)

        document_average_dict["Number of Documents"] = num_docs
        document_headers = ["Score", eval_args.task_name]
        document_report_table = tabulate(
            [[k, v] for k, v in document_average_dict.items()],
            headers=document_headers,
        )

        print(f"\n\nFINAL -- SPLIT LEVEL DOCUMENT AVERAGED RESULTS\n\n")

        print(document_report_table)


if __name__ == "__main__":
    main()
