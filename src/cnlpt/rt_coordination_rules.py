from .pipelines import ctakes_tok
from collections import deque
from itertools import chain, filterfalse, groupby


def unhash_uniq(ls):
    """
    Ugly hack for getting getting unique labels since dictionaries are unhashable

    Args:
      ls: list of items 

    Returns:
      list of unique items from ls
    """
    uniq = []
    for item in ls:
        if item not in uniq:
            uniq.append(item)
    return uniq


def admit_rel(label, src_dose, trg_dose):
    """
    Generates a new relation from label with a classification and a score
    and a new pair of entity mentions to encapsulate with the same labeling information

    Args:
      label: tuple of (first indices, second indices, classification dictionary) 
      src_dose: dose mention in the label
      trg_dose: dose mention to which we want to link the attribute in the label

    Returns:
      new tuple of (first indices, second indices, classification dictionary) 
      with the indices corresponding to src_dose being replaced with the trg_dose  
    """
    sig_span = [*filter(lambda s: s != src_dose, label[:2])][0]
    _, _, sent_dict = label
    new_sent_dict = sent_dict.copy()
    new_sent_dict["source"] = "inference"
    return min(sig_span, trg_dose), max(sig_span, trg_dose), new_sent_dict


def _coordinate_doses(candidate, fixed_labels):
    """
    Coordinates a list of labels across a given DOSE-DOSE pair

    Args:
      candidate: relation tuple of two doses 
      fixed_labels: other labels for which we want to generate new labels based on the candidate dose pair  

    Returns:
      list of original labels plus any new labels from the coordination over the candidate pair
    """
    first_dose, second_dose, _ = candidate

    # checks whether the first dose from the candidate is in the label and not the second dose
    def first_not_second(label):
        inds = label[:2]
        return first_dose in inds and second_dose not in inds

    # checks whether the second dose from the candidate is in the label and not the first dose
    def second_not_first(label):
        inds = label[:2]
        return second_dose in inds and first_dose not in inds

    first_dose_rels = filter(first_not_second, fixed_labels)
    second_dose_rels = filter(second_not_first, fixed_labels)

    # generates new label where the first dose candidate indices
    # have been replaced by the second dose candidate indices
    def first_to_second(label):
        return admit_rel(label, first_dose, second_dose)

    # generates label where the first dose candidate indices
    # have been replaced by the second dose candidate indices
    def second_to_first(label):
        return admit_rel(label, second_dose, first_dose)

    new_seconds = map(first_to_second, first_dose_rels)
    new_firsts = map(second_to_first, second_dose_rels)
    return unhash_uniq([*new_firsts, *new_seconds])


def dose_link_coordination(labels):
    """
    There are a number of linked doses and 
    relations that should arise from the linked doses that 
    the raw system predictions fail to discover.  
    This rule recovers some cases

    Args:
      labels: list of labels 

    Returns:
      original labels with additional new labels from the rule
      (1 , 2 ) : DOSE-X and ( 2 , 3 ) : DOSE-DOSE and ( 3 , 4 ) : DOSE-Y ->
      (1 , 3 ) : DOSE-X and ( 2 , 4 ) DOSE-Y
      We make sure we have transitive closure of DOSE-DOSE instances first 
      before generating the new labels. We add the transitive closure and the resulting 
      labels to the original list
    """

    # checks if a label is DOSE-DOSE
    def linked_doses(t):
        _, _, sent_dict = t
        return sent_dict["label"] == "DOSE-DOSE"

    # IGNORE
    # from an older experiment and does not
    # change anything in the code, will remove
    def to_ground(t):
        first, second, sent_dict = t
        # TESTING
        # sent_dict["ground"] = True
        return first, second, sent_dict

    generator_candidates = [*filter(linked_doses, labels)]

    # get the transitive closure of single given DOSE-DOSE
    # against all the other DOSE-DOSE labels
    def transitive_doses(candidate):
        return _coordinate_doses(candidate, generator_candidates)

    # get the transitive closure of single given DOSE-DOSE
    # against _all_ the other labels (not just DOSE-DOSE)
    def coordinate_doses(candidate):
        return _coordinate_doses(candidate, labels)

    # If more than one DOSE-DOSE
    # compute the transitive closure of the
    # DOSE-DOSEs
    if len(generator_candidates) > 1:

        # here we're just using the naive (non-qualified)
        # maximum to broadcast the signal of the
        # highest scoring DOSE-DOSE
        # intuition:
        # if we trust it enough to keep it we trust it enough to use it
        def best_score(groupby_obj):
            _, group = groupby_obj
            return max([*group], key=lambda t: t[2]["score"])

        # get transitive closure of DOSE-DOSEs, including redundancies
        redundant_candidates = [
            *generator_candidates,
            *chain.from_iterable(map(transitive_doses, generator_candidates)),
        ]

        # filter redundancies
        # i.e. DOSE-DOSEs that share indices
        proto_candidates = [
            *map(
                best_score,
                groupby(
                    sorted(redundant_candidates, key=lambda s: s[:2]),
                    key=lambda t: t[:2],
                ),
            )
        ]
        # ignore this and the analogous statement in the following clause
        # should just be:
        # candidates = proto_candidates
        candidates = [*map(to_ground, proto_candidates)]
    else:
        # candidates = generator_candidates
        candidates = [*map(to_ground, generator_candidates)]

    new_labels = []
    if any(candidates):

        # get the coordinated labels over everything
        # now that we have the transitive closure
        new_labels = [*chain.from_iterable(map(coordinate_doses, candidates))]

    # remove redundancies
    return unhash_uniq([*labels, *new_labels])


def linked_date_coordination(labels, date_map):
    """
    The system's predictive power is less capable on date detection 
    and DOSE-DATE prediction.  This rule captures some missed cases

    Args:
      labels: list of paragraph labels  
      date_map: dictionary of linked date mentions in the paragraph

    Returns:
      labels with new DOSE-DATEs for each date mentions with a dose mention which is associated 
      with one of its linked dates in date_map
    """

    # checks if label is tagged as DOSE-DATE
    def is_date(label):
        _, _, sent_dict = label
        return sent_dict["label"] == "DOSE-DATE"

    candidates = [*filter(is_date, labels)]

    # given a label, a dose mention, and a date mention,
    # generate a new DOSE-DATE label over the dose and date mention
    # spans
    def new_date(sent_dict, dose, date):
        new_sent_dict = sent_dict.copy()
        new_sent_dict["source"] = "inference"
        return min(dose, date), max(dose, date), new_sent_dict

    originals = [*filterfalse(is_date, labels)]

    # given a label, if a label contains a date mention with linked dates,
    # generate new DOSE-DATE labels from the dose mention in the given
    # label and any linked dates
    def generate_dates(label):
        first_span, second_span, sent_dict = label
        l_date = first_span if first_span in date_map.keys() else second_span
        l_dose = first_span if first_span not in date_map.keys() else second_span
        if not any(date_map[l_date]):
            return [label]

        def to_date(candidate):
            return new_date(sent_dict, l_dose, candidate)

        return [label, *map(to_date, date_map[l_date])]

    return [*originals, *chain.from_iterable(map(generate_dates, candidates))]


def boost_site_coordination(paragraph, labels):
    """
    Every DOSE-BOOST admits a DOSE-SITE, 
    but our relation model can only predict one label per instance,
    this rule remedies that limitation

    Args:
      paragraph: paragraph text 
      labels: span labels from model predictions

    Returns:
      labels with recovered DOSE-BOOSTs and a corresponding DOSE-SITE
      for every DOSE-BOOST (on the same span)
    """
    normalized_tokens = [tok.lower() for tok in ctakes_tok(paragraph)]

    # obtain simplified indices
    # e.g. (first_0, first_1) , (second_0, second_1) -> (first_0, second_0)
    # for cnlpt scoring and add them to labels 
    def gen_cnlpt_indices(label):
        first_span, second_span, sent_dict = label
        first_ind, _ = first_span
        second_ind, _ = second_span
        assert first_ind <= second_ind, f"Spans stored and calc'ed incorrectly {label}"
        if "source" not in sent_dict:
            sent_dict["source"] = "prediction"
        sent_dict["cnlpt"] = first_ind, second_ind
        return first_span, second_span, sent_dict

    # checks if span label was tagged as DOSE-BOOST
    def is_boost(label):
        _, _, sent_dict = label
        return sent_dict["label"] == "DOSE-BOOST"

    # generate a new DOSE-SITE from a given label
    def boost_to_site(label):
        first_span, second_span, sent_dict = label
        new_sent_dict = sent_dict.copy()
        cnlpt_inds = sent_dict["cnlpt"]
        first, second = cnlpt_inds
        new_sent_dict["label"] = "DOSE-SITE"
        new_sent_dict["cnlpt"] = second, first
        new_sent_dict["source"] = "inference"
        return first_span, second_span, new_sent_dict

    # check if a label should be a DOSE-BOOST
    # but was mistagged
    def undetected_boost(label):
        first_span, second_span, sent_dict = label
        cnlpt_1, cnlpt_2 = sent_dict["cnlpt"]
        f1, f2 = first_span
        s1, s2 = second_span
        first_text = " ".join(normalized_tokens[f1 : f2 + 1])
        second_text = " ".join(normalized_tokens[s1 : s2 + 1])
        if sent_dict["label"] != "DOSE-SITE" or (cnlpt_1 >= cnlpt_2):
            return False
        undetected_boost = (
            "boost" in first_text.lower() or "boost" in second_text.lower()
        )
        return undetected_boost  

    # make a new DOSE-BOOST out of a given label
    def to_boost(label):
        first_span, second_span, sent_dict = label
        new_sent_dict = sent_dict.copy()
        new_sent_dict["label"] = "DOSE-BOOST"
        new_sent_dict["source"] = "inference"
        return first_span, second_span, new_sent_dict

    cnlpt_info_labels = [*map(gen_cnlpt_indices, labels)]
    new_sites = [*map(boost_to_site, filter(is_boost, cnlpt_info_labels))]
    recovered_boosts = [*map(to_boost, filter(undetected_boost, cnlpt_info_labels))]
    remaining_originals = [*filterfalse(undetected_boost, cnlpt_info_labels)]
    recovered_sites = [*map(boost_to_site, recovered_boosts)]
    return [*recovered_boosts, *new_sites, *recovered_sites, *remaining_originals]


def qual_max(label_ls):
    """
    The model tends to overuse classification as 'None', 
    as a result we have this 'qualified maximum' function as 
    a corrective heuristic

    Args:
      label_ls: List of labels that share the same spans within the sentence 

    Returns:
      The unique label with the 'qualified maximum' score: 
      i.e. prefer non-'None' label to 'None' label even if 'None' is highest score
      (within a difference of 0.25)
    """
    grounds = [*filter(lambda t: t[2]["ground"], label_ls)]
    if any(grounds):
        assert len(grounds) == 1, f"multiple grounds per shared span! {grounds}"
        return grounds[0]
    positive_labels = [*filter(lambda t: t[2]["label"] != "None", label_ls)]
    negative_labels = [*filter(lambda t: t[2]["label"] == "None", label_ls)]
    if not positive_labels:
        return max(negative_labels, key=lambda t: t[2]["score"])
    if not negative_labels:
        return max(positive_labels, key=lambda t: t[2]["score"])
    positive_candidate = max(positive_labels, key=lambda t: t[2]["score"])
    negative_candidate = max(negative_labels, key=lambda t: t[2]["score"])
    if (
        abs(positive_candidate[2]["score"] - negative_candidate[2]["score"]) < 0.25
    ):  # 0.1:
        return positive_candidate
    return negative_candidate


def filter_and_extrapolate_labels(paragraph, labels, date_map):
    """
    Main entry point for the rule-based postprocessing of the raw system predictions 

    Args:
      paragraph: paragraph text 
      labels: labels predicted by the model
      date_map: dictionary of linked dates

    Returns:
      original labels plus additional labels obtained from boost recovery and boost <-> site,
      dose <-> dose, and date <-> date coordination rules. 
    """

    # NOT JUST MAXIMUM SCORE
    # see qual_max docstring
    def best_score(groupby_obj):
        _, group = groupby_obj
        return qual_max([*group])

    # ignore, from an early experiment,
    # to be removed
    def add_ground(label):
        first_span, second_span, sent_dict = label
        sent_dict["ground"] = False
        return first_span, second_span, sent_dict

    proto_ground_labels = [*map(add_ground, labels)]

    
    # obtain simplified indices
    # e.g. (first_0, first_1) , (second_0, second_1) -> (first_0, second_0)
    # for cnlpt scoring 
    def cnlpt_inds(label):
        first_span, second_span, _ = label
        first_ind, _ = first_span
        second_ind, _ = second_span
        return first_ind, second_ind

    proto_coordinated = [
        *map(
            best_score,
            groupby(sorted(proto_ground_labels, key=cnlpt_inds), key=cnlpt_inds),
        )
    ]

    coordinated_labels = dose_link_coordination(proto_coordinated)

    # Remember, groupby is like *nix uniq in that it only works on
    # sorted input
    highest_scores = map(
        best_score,
        groupby(sorted(coordinated_labels, key=cnlpt_inds), key=cnlpt_inds),
    )

    pre_date_coordinated = boost_site_coordination(paragraph, highest_scores)

    fully_coordinated = linked_date_coordination(pre_date_coordinated, date_map)

    return fully_coordinated
