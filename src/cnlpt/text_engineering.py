import operator
import re

from enum import Enum
from heapq import merge

from .pipelines import ctakes_tok
from collections import defaultdict, deque
from itertools import accumulate, combinations, groupby, tee


class ChunkType(Enum):
    """ """
    BEGIN = 0
    HEADING_CR = 1
    TABLE_CR = 2
    BODY_TEXT = 3
    CHUNK_TEXT = 4


def get_chunks(paragraph):
    """
    Get paragraph chunks (pre-windowing) 
    for better detection from dose NER

    Args:
      paragraph: raw paragraph  

    Returns:
      <cr>-parsing chunked paragraph pairs of indices and text, 
     e.g. ( (0, 3) , '<cr> <cr> <cr> <cr>' )
    """

    table_run = {ChunkType.CHUNK_TEXT, ChunkType.TABLE_CR}
    stack = deque()
    stack.append((0, ChunkType.BEGIN))

    # get text chunk type according to specs
    def get_type(is_cr, text_len):
        if is_cr:
            if text_len > 2:
                return ChunkType.HEADING_CR
            else:
                return ChunkType.TABLE_CR
        if text_len > 10:
            return ChunkType.BODY_TEXT
        return ChunkType.CHUNK_TEXT

    def is_cr_cluster(tok):
        return tok.strip() == "<cr>"

    # each text segment gets an index corresponding to the
    # chunk of which it is a member 
    def chunk(type_text_pair):
        counter, prev_type = stack.pop()
        curr_is_cr, curr_text = type_text_pair
        curr_len = len(ctakes_tok(curr_text))
        curr_type = get_type(curr_is_cr, curr_len)
        if prev_type == ChunkType.BEGIN:
            stack.append((counter, curr_type))
            return counter
        if prev_type == ChunkType.HEADING_CR:
            stack.append((counter, curr_type))
            return counter
        if prev_type == ChunkType.BODY_TEXT:
            stack.append((counter + 1, curr_type))
            return counter + 1
        if prev_type in table_run:
            if curr_type in table_run or curr_type == ChunkType.BODY_TEXT:
                stack.append((counter, curr_type))
                return counter
            else:
                stack.append((counter + 1, curr_type))
                return counter + 1

    # From https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/recipes.html 
    # (this is built in to itertools in Python 3.10 but cnlpt v0.3.0 requires 3.8)    
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    # full chunk string reassembled from the chunk tokens
    def resolve_chunk(chunk_grouby_object):
        key, value = chunk_grouby_object
        # [0, text_0] , [1 , text_1] -> [0 , 1] , [text_0 , text_1]
        chunk_indices, strs = zip(*[*value])
        return " ".join(strs)

    cr_split = filter(None, re.split("(\s*<cr>\s*)", paragraph))
    groups = [(k, "".join([*v])) for k, v in groupby(cr_split, is_cr_cluster)]
    text_chunks = [*map(resolve_chunk, groupby(groups, chunk))]
    text_lengths = [len(ctakes_tok(text_chunk)) for text_chunk in text_chunks]
    chunk_indices = [*pairwise(accumulate([0, *text_lengths], operator.add))]
    return zip(chunk_indices, text_chunks)


def noncr_2_cr_inds(chunk):
    """
    Cleans a string of cr tokens, returns the cleaned string
    and the indices of the tokens relative to the original string

    Args:
      chunk: chunk text 

    Returns:
      list of non-cr tokens, list of their indices in the chunk
    """
    tokens = ctakes_tok(chunk)

    def non_cr(tok_pair):
        tok, _ = tok_pair
        return tok.strip() != "<cr>"

    tok_ind_pairs = [*filter(non_cr, zip(tokens, range(0, len(tokens))))]

    if not any(tok_ind_pairs):
        return None, None

    filtered_tokens, filtered_inds = zip(*tok_ind_pairs)
    return " ".join(filtered_tokens), [*filtered_inds]


 
def transitive_closure(iterable):
    """
    Taken from https://stackoverflow.com/a/8674062

    Args:
      iterable: iterable of pairs 

    Returns:
      Transitive closure of iterable, e.g. ( a , b ) , ( b , c ) -> ( a , c ), 
      so set with closure is ( a , b ) , ( b , c ) , ( a , c )
    """
    closure = set(iterable)
    while True:
        new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)

        closure_until_now = closure | new_relations

        if closure_until_now == closure:
            break

        closure = closure_until_now

    return closure


def candidate_date_links(date_indices):
    """
    Finds which dates are possibly linked

    Args:
      date_indices: indices of date mentions in a paragraph 

    Returns:
      pairs of date mentions which are possibly connected (separated by only one token)
    """
    
    date_pairs = combinations(sorted(date_indices), 2)
    # checks whether the dates are separated by a single connector
    def single_connect(date_pair):
        d1, d2 = date_pair
        return (d2[0] - d1[1]) == 1

    return [*filter(single_connect, date_pairs)]


def get_date_links(paragraph, date_indices):
    """
    Builds a table of linked dates:
    give a date, get its linked dates

    Args:
      paragraph: paragraph text 
      date_indices: token indices of date mentions in the paragraph

    Returns:
      Dictionary of linked dates, e.g. if d1, d2, and d3 are linked then:
      d1 -> {d2, d3}
      d2 -> {d1, d3}
      d3 -> {d1, d2}
    """
    link_tokens = {"to", "/", "-"}
    date2links = defaultdict(lambda: set())
    tokens = ctakes_tok(paragraph)
    possible_links = candidate_date_links(date_indices)

    # whether a possibly linked pair of dates
    # is connected by a date linking token
    # (one of 'to', '/', or '-')
    def is_link(p_link):
        first_date, _ = p_link
        _, second = first_date
        return tokens[second + 1].strip().lower() in link_tokens

    all_links = transitive_closure([*filter(is_link, possible_links)])
    for link in all_links:
        d1, d2 = link
        date2links[d1].add(d2)
        date2links[d2].add(d1)
    return date2links


def get_intersect(ls1, ls2):
    """
    Adapted from https://stackoverflow.com/a/57293089
    for getting overlapping intervals from two lists of tuples

    Args:
      ls1: list of intervals/tuples 
      ls2: list of intervals/tuples

    Returns:
      List of intervals/tuples describing intersections between the tuples
      in the two lists, e.g.:
      get_intersect([(2, 4), (9, 12)], [(3, 5), (7, 10)]) -> [(3, 4), (9,10)]
    """
    m1, m2 = tee(merge(ls1, ls2, key=lambda k: k[0]))
    next(m2, None)
    out = []
    for v, g in groupby(zip(m1, m2), lambda k: k[0][1] < k[1][0]):
        if not v:
            ls = [*g][0]
            inf = max(i[0] for i in ls)
            sup = min(i[1] for i in ls)
            # if inf != sup:
            out.append((inf, sup))
    return out
