import os
import re
import torch
import warnings

from .cnlp_processors import (
    cnlp_processors,
    cnlp_output_modes,
    tagging,
    classification,
    classifier_to_relex,
    axis_tags,
    signature_tags,
)

from .pipelines.tagging import TaggingPipeline
from .pipelines.classification import ClassificationPipeline
from .pipelines import ctakes_tok

from .CnlpModelForClassification import CnlpModelForClassification

from .text_engineering import get_chunks, get_date_links, get_intersect, noncr_2_cr_inds
from .rt_coordination_rules import filter_and_extrapolate_labels
from .date_fsm import get_dates

from transformers import AutoConfig, AutoTokenizer


from operator import itemgetter
from collections import Counter
from itertools import chain, zip_longest

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


# Get dictionary of entity tagging models/pipelines
# and relation extraction models/pipelines
# both indexed by task names
def model_dicts(models_dir):
    """

    Args:
      models_dir: directory of <dir name> / <cnlp task name> / pytorch_model.bin (etc)
      batch_size: batch size for the pipelines
    Returns:
      NER model dictionary of cnlp task name -> cnlp model,
      Classification model dictionary of cnlp task name -> cnlp model
    """
    # Pipelines go to CPU (-1) by default so if
    # available send to GPU (0)
    main_device = 0 if torch.cuda.is_available() else -1

    taggers_dict = {}
    out_model_dict = {}

    # For each folder in the model_dir...
    for file in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, file)
        task_name = str(file)
        if os.path.isdir(model_dir) and task_name in cnlp_processors.keys():

            # Load the model, model config, and model tokenizer
            # from the model foldner
            config = AutoConfig.from_pretrained(
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

            # Right now assume roberta thus
            # add_prefix_space = True
            # but want to generalize eventually to
            # other model tokenizers, in particular
            # Flair models for RadOnc

            task_processor = cnlp_processors[task_name]()

            # Add tagging pipelines to the tagging dictionary
            if cnlp_output_modes[task_name] == tagging:
                taggers_dict[task_name] = TaggingPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task_processor=task_processor,
                    device=main_device,
                    # Just get rid of it
                    # batch_size=batch_size,
                )
            # Add classification pipelines to the classification dictionary
            elif cnlp_output_modes[task_name] == classification:
                # now doing return_all_scores
                # no matter what
                classifier = ClassificationPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    return_all_scores=True,
                    task_processor=task_processor,
                    device=main_device,
                    # this is done one by one anyway...
                    # batch_size=batch_size,
                )
                out_model_dict[task_name] = classifier
            # Tasks other than tagging and sentence/relation classification
            # not supported for now since I wasn't sure how to fit them in
            else:
                ValueError(
                    (
                        "output mode "
                        f"{cnlp_output_modes[task_name]}"
                        "not currently supported"
                    )
                )
    return taggers_dict, out_model_dict


def generate_paragraph_casoids(paragraphs, taggers_dict, axis_task):
    """

    Args:
      paragraphs: list of plaintext cTAKES tokenized paragraphs
      taggers_dict: NER model dictionary
      axis_task: NER task which is the anchor of all the relations

    Returns:
      list (iterable map object) of CAS-like (as in UIMA CAS) objects,
      one per paragraph
    """

    def process(paragraph):
        """

        Args:
          paragraph:

        Returns:
          Paragraph CASoid with NER mentions and annotated text windows
        """
        return window_assemble(paragraph, taggers_dict, axis_task)

    return map(process, paragraphs)


def window_assemble(paragraph, taggers_dict, axis_task):
    """

    Args:
      paragraphs: plaintext cTAKES tokenized paragraph
      taggers_dict: NER model dictionary
      axis_task: NER task which is the anchor of all the relations

    Returns:
      Paragraph and paragraph CASoid with NER mentions and annotated text windows
    """
    return paragraph, get_window_dictionary(paragraph, taggers_dict, axis_task)


def get_window_dictionary(paragraph, taggers_dict, axis_task):
    """

    Args:
      paragraphs: plaintext cTAKES tokenized paragraph
      taggers_dict: NER model dictionary
      axis_task: NER task which is the anchor of all the relations

    Returns:
      Paragraph CASoid with NER mentions and annotated text windows


    """

    raw_dose_inds = get_dose_indices(paragraph, taggers_dict, axis_task)

    paragraph_chunk_dose_pairs = [*filter(None, raw_dose_inds)]

    def local_w_indices(chunk_dose_pair):
        """

        Args:
          chunk_dose_pair indices of <cr>-parsing based text chunk, list of dose indices in the chunk

        Returns:
          Windows within the chunk centered around each of the dose mentions
        """
        chunk_indices, dose_indices = chunk_dose_pair
        return get_window_indices(chunk_indices, dose_indices)

    def local_window_dict(chunk_dose_pair):
        """

        Args:
          chunk_dose_pair: indices of <cr>-parsing based text chunk, list of dose indices in the chunk

        Returns:
          Nested dictioary of annotated windows in the paragraph, dose mention indices are the top level keys
        """
        other_doses = [
            dose_inds
            for _, dose_inds in paragraph_chunk_dose_pairs
            if (_, dose_inds) != chunk_dose_pair
        ]
        return get_annotated_window_dict(
            chunk_dose_pair,
            other_doses,
            paragraph,
            taggers_dict,
            axis_task,
        )

    return {
        local_w_indices(dose_indices): local_window_dict(dose_indices)
        for dose_indices in paragraph_chunk_dose_pairs
    }


def get_window_indices(chunk_indices, central_dose_indices):
    """

    Args:
      chunk_indices: beginning index of chunk in paragraph, end index of chunk in paragraph
      central_dose_indices: paragraph level indices of the dose mention used to define the window center

    Returns:
      Paragraph level indices of the window around the dose mention, dose indices
    """
    chunk_start, chunk_end = chunk_indices
    first, second = central_dose_indices
    window_size = 53  # x2 = 106 for full span
    center_index = ((second - first) // 2) + second
    # this accounts for the window being possibly the entire paragraph

    begin_diff = center_index - window_size
    end_diff = center_index + window_size + 1

    w_start = max(chunk_start, begin_diff)
    w_end = min(chunk_end, end_diff)  # bc Python list slices need one extra
    # this accounts for the case where the whole paragraph
    # is shorter than a normal window but has multiple doses
    # for which we want to run the model
    return (w_start, w_end), central_dose_indices


def get_dose_indices(paragraph, taggers_dict, axis_task):
    """

    Args:
      paragraphs: plaintext cTAKES tokenized paragraph
      taggers_dict: NER model dictionary
      axis_task: NER task which is the anchor of all the relations
    Returns:
      Iterable of dose indices in the paragraph
    """

    def get_chunk_doses(chunk_indices, chunk):
        """

        Args:
          chunk_indices: beginning index of chunk in paragraph, end index of chunk in paragraph
          chunk: actual chunk text

        Returns:
          List of indices of dose mentions within the chunk
        """
        return get_chunk_dose_indices(chunk_indices, chunk, taggers_dict, axis_task)

    return chain.from_iterable(
        get_chunk_doses(chunk_indices, chunk)
        for chunk_indices, chunk in get_chunks(paragraph)
    )


def get_chunk_dose_indices(chunk_indices, chunk, taggers_dict, axis_task):
    """

    Args:
      chunk_indices:
      chunk:
      taggers_dict:
      axis_task:

    Returns:

    """
    chunk_start, chunk_end = chunk_indices
    dose_model = taggers_dict[axis_task]
    filtered_chunk, filtered_inds = noncr_2_cr_inds(chunk)
    if filtered_chunk is None:
        return []
    chunk_ann = dose_model(filtered_chunk)

    """
    print(f"chunk:\n{chunk}\n")
    print(f"filtered chunk:\n{filtered_chunk}")
    print(f"filtered indices:\n{filtered_inds}\n")
    print(f"model annotation:\n{chunk_ann}\n")
    """    
    raw_dose_chunk_indices = process_ann(chunk_ann)

    # print(f"dose indices from model:\n{raw_dose_chunk_indices}\n")
    
    dose_chunk_indices = [
        itemgetter(*dose_inds)(filtered_inds) for dose_inds in raw_dose_chunk_indices
    ]

    return [
        ((chunk_start, chunk_end), (chunk_start + dose_start, chunk_start + dose_end))
        for dose_start, dose_end in dose_chunk_indices
    ]


def get_annotated_window_dict(
    chunk_dose_pair, paragraph_dose_indices, paragraph, taggers_dict, axis_task
):
    """

    Args:
      chunk_dose_pair: chunk indices and selected centraldose indices for the window
      paragraph_dose_indices: non central dose indices in the paragraph
      paragraph: paragraph text
      taggers_dict: dictionary of NER taggers
      axis_task: dose task in this case

    Returns:

    """
    chunk_indices, dose_indices = chunk_dose_pair
    task_2_indices = {}

    window_indices, _ = get_window_indices(chunk_indices, dose_indices)

    window_start, window_end = window_indices

    non_central_dose_indices = get_non_central_indices(
        paragraph_dose_indices, window_start, window_end
    )
    task_2_indices["rt_dose"] = non_central_dose_indices
    window_tokens = ctakes_tok(paragraph)[window_start:window_end]
    raw_window = " ".join(window_tokens)
    filtered_window, filtered_inds = noncr_2_cr_inds(raw_window)
    # regex_dates = {"regex_dates": get_dates(filtered_window)}

    def recast_inds(model_out):
        """

        Args:
          model_out: output string from the NER model

        Returns
          Cleaned (no <cr>) window relative indices of any mentions discovered by the NER model
        """
        sig_ann = process_ann(model_out)

        def recast(inds):
            """

            Args:
              inds: Non cleaned (with <cr>) window relative indices

            Returns:
              Cleaned (no <cr>) window relative indices
            """
            return itemgetter(*inds)(filtered_inds)

        return [*map(recast, sig_ann)]

    sig_2_indices = {}

    if filtered_window is not None:
        sig_2_indices = {
            sig_task: recast_inds(sig_model(filtered_window))
            for sig_task, sig_model in taggers_dict.items()
            if sig_task != axis_task
        }
    task_2_indices.update(sig_2_indices)
    # task_2_indices.update(regex_dates)
    return get_merged_annotation_dict(
        dose_indices, task_2_indices, axis_task, raw_window, (window_start, window_end)
    )


def get_merged_annotation_dict(
    dose_indices, task_2_indices, axis_task, window, window_indices
):
    """

    Args:
      dose_indices: central dose indices in the window
      task_2_indices: dictionary of NER task name -> window relative indices of NER task mentions
      axis_task: central NER task (rt_dose)
      window: window text
      window_indice: paragraph relative window indices

    Returns:

    """

    def local_merge(sig_span, sig_task):
        """

        Args:
          sig_span: window relative attribute mention indices
          sig_task: attribute's NER task name

        Returns:
          Annotated window with tags around the central dose and around the provided attribute
        """
        return merge_annotations(
            dose_indices, axis_task, sig_span, sig_task, window, window_indices
        )

    def get_sig_index_dict(sig_indices, sig_task):
        """

        Args:
          sig_indices: list of attribute indices
          sig_task: attribute's NER task name

        Returns:
          Dictionary of attribute indices -> annotated window for central dose mention and attribute mention
        """
        return {s_inds: local_merge(s_inds, sig_task) for s_inds in sig_indices}

    merged_annotations = {
        task: get_sig_index_dict(indices, task)
        for task, indices in task_2_indices.items()
    }
    return merged_annotations


def merge_annotations(axis_span, axis_task, sig_span, sig_task, window, window_indices):
    """

    Args:
      axis_span: central dose mention window relative indices
      axis_task: axis mention type (dose) NER task name (rt_dose)
      sig_span: attribute mention window relative indices
      sig_task: attribute's NER task name
      window: window text
      window_indices: paragraph relative window indices

    Returns:
      Annotated window with tags around the central dose and around the provided attribute
    """
    axis_tag_begin, axis_tag_close = axis_tags[axis_task]
    sig_tag_begin, sig_tag_close = signature_tags[sig_task]

    window_start, _ = window_indices
    _a1, _a2 = axis_span
    a1, a2 = _a1 - window_start, _a2 - window_start
    ref_sent = ctakes_tok(window)
    ann_sent = ref_sent.copy()
    ann_sent[a1] = axis_tag_begin + " " + ann_sent[a1]
    ann_sent[a2] = ann_sent[a2] + " " + axis_tag_close

    s1, s2 = sig_span

    intersects = get_intersect([(a1, a2)], [(s1, s2)])
    if intersects:
        warnings.warn(
            (
                f"Warning axis annotation and sig annotation \n"
                f"{ref_sent}\n"
                f"{a1, a2}\n"
                f"{s1, s2}\n"
                f"Have intersections at span:\n"
                f"{intersects}"
            )
        )

    ann_sent[s1] = sig_tag_begin + " " + ref_sent[s1]
    ann_sent[s2] = ref_sent[s2] + " " + sig_tag_close
    return {"ann_window": " ".join(ann_sent)}


def get_non_central_indices(paragraph_dose_indices, window_start, window_end):
    """

    Args:
      paragraph_dose_indices: paragraph relative indices of all doses other than this window's central dose
      window_start: paragraph relative window beginning
      window_end: paragraph relative window ending

    Returns:

    """

    def window_adjust(entity_span):
        """

        Args:
          entity_span: paragraph relative span indices

        Returns:
          window relative entity span
        """
        entity_start, entity_end = entity_span
        # since 0 is first span relative span idx and
        # w_end - w_begin is the second
        return (
            max(entity_start - window_start, 0),
            min(entity_end - window_start, window_end - window_start),
        )

    def in_window(entity_span):
        """

        Args:
          entity_span: paragraph relative span indices

        Returns:
          Whether the entity span at least partially overlaps with the window
        """
        return any(i in range(window_start, window_end) for i in entity_span)

    return [*map(window_adjust, filter(in_window, paragraph_dose_indices))]


def get_partitions(annotation):
    """

    Args:
      annotation: NER model output tags

    Returns:
      NER model output tags without NER task info (e.g. B-fxno -> B)
    """
    return "".join(map(lambda tag: tag[0], annotation))


def process_ann(annotation):
    """

    Args:
      annotation:  NER model output tags

    Returns:
      Annotation relative indices of discovered entity mentions
    """
    span_begin, span_end = 0, 0
    indices = []
    partitions = get_partitions(annotation)
    # Group B's individually as well as B's followed by
    # any nummber of I's, e.g.
    # OOOOOOBBBBBBBIIIIBIBIBI
    # -> OOOOOO B B B B B B BIIII BI BI BI
    for span in filter(None, re.split(r"(BI*)", partitions)):
        span_end = len(span) + span_begin - 1
        if span[0] == "B":
            # Get indices in list/string of each span
            # which describes a mention
            indices.append((span_begin, span_end))
        span_begin = span_end + 1
    return indices


def get_sentences_and_labels(in_file: str, mode: str, task_names):
    """

    Args:
      str: in_file:  file containing instances separated by newlines
      str: mode: specifying instance generation for inference or evaluation
      task_names: cnlpt task name which provides schema for instance generation

    Returns:
      Dictionary of task name -> list of task labels,
      one per instance, raw instances,
      number of tokens in the largest instance
    """

    def relex_proc(task_name):
        """

        Args:
          task_name: classification model task name

        Returns:
          Task processor associated with the classification model
          but for end to end metrics
        """
        return cnlp_processors[classifier_to_relex[task_name]]()

    task_processors = [*map(relex_proc, task_names)]
    idx_labels_dict = {}
    if mode == "inf":
        # 'test' lets us forget labels
        # just use the first task processor since
        # _create_examples and _read_tsv are task/label agnostic
        examples = task_processors[0]._create_examples(
            task_processors[0]._read_tsv(in_file), "test"
        )
    elif mode == "eval":
        # 'dev' lets us get labels without running into issues of downsampling
        lines = task_processors[0]._read_tsv(in_file)
        examples = task_processors[0]._create_examples(lines, "dev")
        for example in examples:
            print(example)
        
        def example2label(example):
            """

            Args:
              example: instance object containing raw sentence and label data

            Returns:
              Label data formatted for scoring
            """
            # Just assumed relex
            def conv_tuple(l_tuple):
                """

                Args:
                  l_tuple: idx_1_str, idx_2_str, label_str

                Returns:
                  idx_1, idx_2, label_idx
                """
                (start_token, end_token, category) = l_tuple

                return (int(start_token), int(end_token), label_map.get(category, 0))

            if example.label:
                return [*map(conv_tuple, example.label)]
            else:
                return []

        for task_name, task_processor in zip(task_names, task_processors):
            label_list = task_processor.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}

            # Adjusted for 'relex'
            if examples[0].label is not None:
                # can leave this as a map object since we
                # loop over it when generating the
                # label/ground truth matrices
                # for relex eval
                idx_labels_dict[task_name] = map(example2label, examples)
            else:
                ValueError("labels required for eval mode")
    else:
        ValueError("Mode must be either inference or eval")

    max_len = -1

    def get_sent_len(sent):
        """

        Args:
          sent: raw sentence

        Returns:
          Number of tokens in sentence
        """
        return len(ctakes_tok(sent))

    if examples[0].text_b is None:
        sentences = [example.text_a for example in examples]
        max_len = get_sent_len(max(sentences, key=get_sent_len))
    else:
        sentences = [(example.text_a, example.text_b) for example in examples]

    return idx_labels_dict, sentences, max_len


def classify_casoid_annotations(casoid, out_model_dict):
    """

    Args:
      casoid: Paragraph CASoid with annotated windows
      out_model_dict: dictionary of classification models (task name -> classifier)

    Returns:
      CASoid with each annotated window classified for a relation,
      with its model score as well as the type of attribute NER which
      discovered the non-dose mention
    """
    for w_d_inds, w_dict in casoid.items():
        for task, indcs_dict in w_dict.items():
            for indcs, sent_dict in indcs_dict.items():
                for out_task, out_model in out_model_dict.items():
                    filtered_window, _ = noncr_2_cr_inds(sent_dict["ann_window"])
                    model_output = out_model(
                            filtered_window,
                            padding="max_length",
                            truncation=True,
                            is_split_into_words=True,
                        )
                    # For the dose/attr pair
                    # pick the label with the strongest signal
                    strongest_label_dict = max(
                        # model_output[0],
                        model_output,
                        key=lambda d: d["score"],
                    )

                    best_label = strongest_label_dict["label"]
                    best_score = strongest_label_dict["score"]
                    sent_dict["label_dict"] = {
                        "label": best_label,
                        "score": best_score,
                        "sig_model_src": task,
                    }
                    print(f"Raw: {w_d_inds}\t{indcs}\t{model_output}")
                    print(f"final: {sent_dict['label_dict']}")
    return casoid


def get_rel_indices(dose_indices, sig_indices, window_indices):
    """

    Args:
      dose_indices: paragraph relative dose indices
      sig_indices: window relative attribute indices
      window_indices: window indices

    Returns:
      paragraph level dose and attribute indices,
      ordered by whichever's first element is earliest
      in the paragraph
    """
    s1, s2 = sig_indices
    window_start, _ = window_indices
    paragraph_sig = window_start + s1, window_start + s2
    return min(dose_indices, paragraph_sig), max(dose_indices, paragraph_sig)


def adjust_date_inds(wd2dict):
    """

    Args:
      wd2dict: dictionary of window relative date indices -> annotated windows

    Returns:
      paragraph level date indices
    """
    w_d_inds, w_dict = wd2dict
    w_inds, _ = w_d_inds
    window_start, _ = w_inds

    def adjust(date_inds):
        """

        Args:
          date_inds: window relative date indices

        Returns:
          Paragraph relative date indices
        """
        first, second = date_inds
        return window_start + first, window_start + second

    # Guard for the HemOnc demo
    # where we didn't train a date model
    # since HemOnc has no date instances
    if "rt_date" not in w_dict:
        return []
    return [*map(adjust, w_dict["rt_date"].keys())]


def get_paragraph_dates(casoid):
    """

    Args:
      casoid: paragraph CASoid

    Returns:
      List of dates in the paragraph
    """
    return [*chain.from_iterable(map(adjust_date_inds, casoid.items()))]


def casoid_to_label_tuples(paragraph, casoid):
    """

    Args:
      paragraph: paragraph text
      casoid: paragraph CASoid (with classified windows)

    Returns:
      List of (first span indices, second span indices, relation classification data dictionary)
    """
    labels = []
    for w_d_inds, w_dict in casoid.items():
        for task, indcs_dict in w_dict.items():
            for indcs, sent_dict in indcs_dict.items():
                w_inds, dose_indices = w_d_inds
                sig_indices = indcs
                first_span, second_span = get_rel_indices(
                    dose_indices, sig_indices, w_inds
                )
                labels.append((first_span, second_span, sent_dict["label_dict"]))
    date_map = get_date_links(paragraph, get_paragraph_dates(casoid))
    return filter_and_extrapolate_labels(paragraph, labels, date_map)


def get_entity_label_str(paragraph, label_tuples):
    """

    Args:
      paragraph: paragraph text
      label_tuples: list of (first span indices, second span indices, relation classification data dictionary)

    Returns:
      Output string of '( first index of first span, first index of second span , relation label ) -> ( first entity , second entity )'
      organized by predicted by the model vs inferred from rules
    """
    tokenized_paragraph = ctakes_tok(paragraph)

    def label_to_str(label):
        """

        Args:
          label: (first span indices, second span indices, relation classification data dictionary)

        Returns:
          string of '( first index of first span, first index of second span , relation label ) -> ( first entity , second entity )'
        """
        first_span, second_span, sent_dict = label
        f1, f2 = first_span
        s1, s2 = second_span
        first_entity = " ".join(tokenized_paragraph[f1 : f2 + 1])
        second_entity = " ".join(tokenized_paragraph[s1 : s2 + 1])
        first_ind, second_ind = sent_dict["cnlpt"]
        cnlpt_label = f"( {first_ind} , {second_ind} , {sent_dict['label']})"
        entities = f"( {first_entity} , {second_entity} )"
        return f"{cnlpt_label} -> {entities}"

    def is_pos(label):
        """

        Args:
          label: (first span indices, second span indices, relation classification data dictionary)

        Returns:
          Whether model classified the pair with a non-None label
        """
        _, _, sent_dict = label
        return sent_dict["label"] != "None"

    def is_pred(label):
        """

        Args:
          label: (first span indices, second span indices, relation classification data dictionary)

        Returns:
          Whether the label is a direct result of model prediction (not a rule)
        """

        _, _, sent_dict = label
        return sent_dict["source"] == "prediction"

    def is_inf(label):
        """

        Args:
          label: (first span indices, second span indices, relation classification data dictionary)

        Returns:
          Whether the label is a direct result of a rule applied to model predictions
        """
        _, _, sent_dict = label
        return sent_dict["source"] == "inference"

    def cnlpt_inds(label):
        """

        Args:
          label: (first span indices, second span indices, relation classification data dictionary)

        Returns:
          (first index of first span, first index of second span)
        """
        _, _, sent_dict = label
        return sent_dict["cnlpt"]

    predicted_labels = map(
        label_to_str,
        sorted(filter(is_pred, filter(is_pos, label_tuples)), key=cnlpt_inds),
    )
    inferred_labels = map(
        label_to_str,
        sorted(filter(is_inf, filter(is_pos, label_tuples)), key=cnlpt_inds),
    )

    predicted_str = "Model Predicted Labels:\n\n" + "\n".join(predicted_labels) + "\n\n"
    inferred_str = (
        "Labels Inferred From Predictions:\n\n" + "\n".join(inferred_labels) + "\n\n"
    )
    return predicted_str + inferred_str


def get_entity_columns_str(paragraph, casoid):
    """

    Args:
      paragraph: paragraph text
      casoid: paragraph CASoid

    Returns:
      String of detected dose and attribute mentions in the paragraph,
      organized by column
    """
    tokenized_paragraph = ctakes_tok(paragraph)
    axis_spans = set()
    sig_spans = set()
    for w_d_inds, w_dict in casoid.items():
        for task, indcs_dict in w_dict.items():
            for indcs, sent_dict in indcs_dict.items():
                w_inds, dose_indices = w_d_inds
                sig_indices = indcs
                s1, s2 = sig_indices
                window_start, _ = w_inds
                paragraph_sig = window_start + s1, window_start + s2
                axis_spans.add((*dose_indices, "rt_dose"))
                # to avoid redundancy, mention to Guergana and
                # Danielle when debugging
                if task != "rt_dose":
                    sig_spans.add((*paragraph_sig, task))

    def span_to_entity(span_model):
        """

        Args:
          span_model: idx_1, idx_2, model name

        Returns:
          idx_1, idx_2, entity text, model name
        """
        s1, s2, model_name = span_model
        entity_text = " ".join(tokenized_paragraph[s1 : s2 + 1])
        return s1, s2, entity_text, model_name

    axis_entities = map(span_to_entity, sorted(axis_spans))
    sig_entities = map(span_to_entity, sorted(sig_spans))

    raw_col = [*zip_longest(axis_entities, sig_entities, fillvalue="\t\t")]

    def entity_pair_str(entity_pair):
        """

        Args:
          entity_pair: (idx_1, idx_2, dose string, model name) , (idx_1, idx_2, attr string, model name)

        Returns:
          '<dose string> \t <attr string>'
        """
        axis_entity, sig_entity = entity_pair

        if isinstance(axis_entity, str):
            axis_str = axis_entity
        else:
            a1, a2, a_str, model_name = axis_entity
            axis_str = f"( {a1} , {a2} , {a_str}, source : {model_name} )"

        if isinstance(sig_entity, str):
            sig_str = sig_entity
        else:
            s1, s2, s_str, model_name = sig_entity
            sig_str = f"( {s1} , {s2} , {s_str}, source : {model_name} )"

        return f"{axis_str}\t{sig_str}"

    entity_pairs_str = "\n".join(map(entity_pair_str, raw_col))
    return "Anchor mentions:\tSignature mentions:\n\n" + entity_pairs_str + "\n\n"


def get_relation_counts_str(label_tuples):
    """

    Args:
      label_tuples: List of (first span indices, second span indices, relation classification data dictionary)

    Returns:

    """

    def get_label(label_tuple):
        """

        Args:
          label_tuple: (first span indices, second span indices, relation classification data dictionary)

        Returns:
          relation label
        """
        _, _, sent_dict = label_tuple
        return sent_dict["label"]

    positive_rels = filter(lambda s: s != "None", map(get_label, label_tuples))

    relation_counts = Counter(positive_rels)

    def count_tuple(pair):
        """

        Args:
          pair: label, number of occurences

        Returns:
          '( <label> , <number of occurences> )'
        """
        label, count = pair
        return f"( {label} , {count} )"

    return " , ".join(
        map(
            count_tuple,
            sorted(relation_counts.items(), key=lambda t: t[1], reverse=True),
        )
    )


def casoid_entity_print(parent_dir, filename, paragraph, casoid, label_tuples, idx):
    """

    Args:
      parent_dir: Dir to write the file
      filename: Filename to write to within the dir
      paragraph: paragraph text
      casoid: paragraph CASoid
      label_tuples: List of (first span indices, second span indices, relation classification data dictionary)
      idx: file index within the directory

    Returns:
      None
    """
    positive_label_str = get_entity_label_str(paragraph, label_tuples)
    discovered_entity_table = get_entity_columns_str(paragraph, casoid)
    relation_counts_str = get_relation_counts_str(label_tuples)
    pwd = os.getcwd()
    note_identifier = ".".join(os.path.basename(filename).split(".")[:-1])
    note_dir = os.path.join(pwd, parent_dir + "/" + note_identifier)
    if not os.path.exists(note_dir):
        os.makedirs(note_dir)

    out_fn = "".join(
        [
            "predictions_" + note_identifier,
            ".txt",
        ]
    )
    out_file = os.path.join(note_dir, out_fn)
    with open(out_file, "at") as out_writer:
        out_writer.write(f"{idx + 1}.\n\n")
        out_writer.write(f"Relation type counts:\n{relation_counts_str}")
        out_writer.write("\n\n")
        out_writer.write(f"Predicted positive labels:\n\n{positive_label_str}")
        out_writer.write("\n\n")
        out_writer.write(f"Ground truth paragraph:\n\n{paragraph}")
        out_writer.write("\n\n")
        out_writer.write("Discovered Entities\n\n")
        out_writer.write(f"{discovered_entity_table}\n\n")


def get_cnlpt_labels(labels):
    """

    Args:
      labels: List of (first span indices, second span indices, relation classification data dictionary)

    Returns:
      List of (idx_1, idx_2, label)
    """

    def _label(label):
        """

        Args:
          label: (first span indices, second span indices, relation classification data dictionary)

        Returns:
          (idx_1, idx_2, label)
        """
        _, _, sent_dict = label
        return *sent_dict["cnlpt"], sent_dict["label"]

    return [*map(_label, labels)]


def get_predictions(
    out_dir, in_file, paragraphs, taggers_dict, out_model_dict, axis_task
):
    """

    Args:
      out_dir: output directory
      in_file: source file of instances
      paragraphs: list of paragraphs from in file
      taggers_dict: dictionary of NER taggers, task name -> model
      out_model_dict: dictionary of classifiers, task name -> model
      axis_task: central task (rt_dose)

    Returns:
      List of (paragraph length, paragraph labels of form (idx_1, idx_2, label))
    """

    def classify_casoid(casoid_pair):
        """

        Args:
          casoid_pair: paragraph and its CASoid

        Returns:
         paragraph, CASoid with classified annotation windows
        """
        paragraph, casoid = casoid_pair
        print(paragraph)
        return paragraph, classify_casoid_annotations(casoid, out_model_dict)

    def get_casoid_label(casoid_pair):
        """

        Args:
          casoid_pair: paragraph and its CASoid

        Returns:
          List of (first span indices, second span indices, relation classification data dictionary)
        """
        paragraph, casoid = casoid_pair
        return casoid_to_label_tuples(paragraph, casoid)

    def entity_print(indexed_casoid_pair):
        """

        Args:
          indexed_casoid_pair: index of the paragraph in the document and the paragraph+CASoid pair

        Returns:
          None (printed entity columns)
        """
        idx, casoid_pair = indexed_casoid_pair
        para2casoid, label = casoid_pair
        paragraph, casoid = para2casoid
        return casoid_entity_print(out_dir, in_file, paragraph, casoid, label, idx)

    paragraphs_2_raw_casoids = generate_paragraph_casoids(
        paragraphs, taggers_dict, axis_task
    )

    paragraphs_2_classified_cassoids = [*map(classify_casoid, paragraphs_2_raw_casoids)]

    casoid_labels = [*map(get_casoid_label, paragraphs_2_classified_cassoids)]

    casoid_proto_entities = enumerate(
        zip(paragraphs_2_classified_cassoids, casoid_labels)
    )
    _ = [*map(entity_print, casoid_proto_entities)]

    def proto_matrix(pl_pair):
        """

        Args:
          pl_pair: paragraph and its raw label tuples

        Returns:
          paragraph length and cnlpt labels
        """
        paragraph, labels = pl_pair
        paragraph_length = len(ctakes_tok(paragraph))
        cnlpt_labels = get_cnlpt_labels(labels)
        return paragraph_length, cnlpt_labels

    return [*map(proto_matrix, zip(paragraphs, casoid_labels))]
