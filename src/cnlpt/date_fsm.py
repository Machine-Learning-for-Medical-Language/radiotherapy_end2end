import re

# from .pipelines import ctakes_tok
from collections import deque
from transitions import Machine
from itertools import chain

# ipython
def ctakes_tok(s):
    """

    Args:
      s: 

    Returns:

    """
    return [*filter(None, s.split())]


def clean_numeric(t):
    """

    Args:
      t: 

    Returns:

    """
    if t[:-1].isnumeric() and not t[-1].isnumeric():
        return t[:-1]
    return t


def possible_year(t):
    """

    Args:
      t: 

    Returns:

    """
    cleaned_token = clean_numeric(t)
    return cleaned_token.isnumeric() and int(cleaned_token) in range(1, 3000)


def possible_day(t):
    """

    Args:
      t: 

    Returns:

    """
    cleaned_token = clean_numeric(t)
    return cleaned_token.isnumeric() and int(cleaned_token) in range(1, 32)


def possible_month(t):
    """

    Args:
      t: 

    Returns:

    """
    cleaned_token = clean_numeric(t)
    return cleaned_token.isnumeric() and int(cleaned_token) in range(1, 13)


def possible_year_not_day(t):
    """

    Args:
      t: 

    Returns:

    """
    cleaned_token = clean_numeric(t)
    return cleaned_token.isnumeric() and int(cleaned_token) in range(32, 3000)


def parse_numerics(s_toks, numeric_inds):
    """

    Args:
      s_toks: 
      numeric_inds: 

    Returns:

    """
    n_1, n_2 = numeric_inds
    candidate = " ".join(s_toks[n_1 : n_2 + 1])
    candidate_spans = re.split("(-)", candidate)
    if len(candidate_spans) != 3:
        return [(n_1, n_2)]
    first_date, _, second_date = candidate_spans
    f1, f2 = (n_1, n_1 + len(first_date.split()) - 1)
    s1, s2 = (n_2 - (len(second_date.split()) - 1), n_2)

    return [(f1, f2), (s1, s2)]


full_months = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

short_months = {
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}


class NumericDateDetector(object):
    """ """
    states = [
        "start",
        "end",
        "nt_end",
        "month_num",
        "month_day_sep",
        "day_num",
        "day_year_sep",
        "fslash_start",
        "fslash_month_num",
        "fslash_month_day_sep",
        "fslash_day_num",
        "fslash_day_year_sep",
        "year_num",
    ]

    def __init__(self):
        self.stack = deque()
        self.indices = set()
        self.machine = Machine(
            model=self, states=NumericDateDetector.states, initial="start"
        )
        self.machine.add_transition(trigger="reset", source="*", dest="start")
        # initial
        self.machine.add_transition(
            trigger="month_num_cond", source="start", dest="month_num"
        )
        self.machine.add_transition(
            trigger="month_num_cond", source="fslash_start", dest="fslash_month_num"
        )

        # first month
        self.machine.add_transition(
            trigger="m_fslash_cond", source="month_num", dest="fslash_month_day_sep"
        )
        self.machine.add_transition(
            trigger="m_fslash_cond",
            source="fslash_month_num",
            dest="fslash_month_day_sep",
        )
        self.machine.add_transition(
            trigger="m_dash_cond", source="month_num", dest="month_day_sep"
        )

        # month-day separator
        self.machine.add_transition(
            trigger="day_num_cond", source="month_day_sep", dest="day_num"
        )
        self.machine.add_transition(
            trigger="day_num_cond", source="fslash_month_day_sep", dest="fslash_day_num"
        )
        # day number
        self.machine.add_transition(
            # either way we're just waiting for the year number so no longer separator sensitive
            trigger="d_fslash_cond",
            source="fslash_day_num",
            dest="day_year_sep",
        )
        self.machine.add_transition(
            trigger="d_dash_cond", source="day_num", dest="day_year_sep"
        )
        self.machine.add_transition(
            trigger="d_dash_cond", source="fslash_day_num", dest="fslash_start"
        )
        self.machine.add_transition(
            trigger="non_separator", source="day_num", dest="nt_end"
        )
        # day-year separator
        self.machine.add_transition(
            trigger="year_num_cond", source="day_year_sep", dest="end"
        )

    def process_token(self, token, token_idx):
        """

        Args:
          token: 
          token_idx: 

        Returns:

        """
        if self.state == "start":
            if possible_month(token):
                self.month_num_cond()
                self.stack.append(token_idx)
            else:
                self.reset()
        elif self.state == "fslash_start":
            if possible_month(token):
                self.month_num_cond()
            else:
                self.reset()
        elif self.state == "month_num":
            if token == "/":
                self.m_fslash_cond()
            elif token == "-":
                self.m_dash_cond()
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "fslash_month_num":
            if token == "/":
                self.m_fslash_cond()
            else:
                self.stack.clear()
                self.reset()
        # does the right transfer state with the same information either way
        elif self.state == "month_day_sep" or self.state == "fslash_month_day_sep":
            if possible_day(token):
                self.day_num_cond()
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "day_num":
            if token == "-":
                self.d_dash_cond()
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "fslash_day_num":
            if token == "-":
                self.d_dash_cond()
            elif token == "/":
                self.d_fslash_cond()
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "day_year_sep":
            if possible_year(token):
                self.year_num_cond()
                # self.stack.append(token_idx)
            else:
                self.stack.clear()
                self.reset()

        if self.state == "end":
            if len(self.stack) == 1:
                first = self.stack.pop()
                second = token_idx
                self.indices.add((first, second))
                self.stack.clear()
                self.reset()
            else:
                print(f"Error! In end state and stack contents : {self.stack}")
        if self.state == "nt_end":
            self.stack.clear()
            self.reset()

    def process_tokens(self, token_ls):
        """

        Args:
          token_ls: 

        Returns:

        """
        for idx, token in enumerate(token_ls):
            self.process_token(token.strip().lower(), idx)
        return self.indices


class TextDateDetector(object):
    """ """
    states = [
        "start",
        "end",
        "nt_end",
        "month_full_text",
        "month_short_text",
        "day_num",
        "comma",
        "period",
    ]

    def __init__(self):
        self.stack = deque()
        self.indices = set()
        self.machine = Machine(
            model=self, states=TextDateDetector.states, initial="start"
        )
        self.machine.add_transition(trigger="reset", source="*", dest="start")
        # Start
        self.machine.add_transition(
            trigger="month_full_cond", source="start", dest="month_full_text"
        )
        self.machine.add_transition(
            trigger="month_short_cond", source="start", dest="month_short_text"
        )
        # Full month
        self.machine.add_transition(
            trigger="day_num_cond", source="month_full_text", dest="day_num"
        )
        self.machine.add_transition(
            trigger="year_not_day_cond", source="month_full_text", dest="end"
        )
        # Shorthand month
        self.machine.add_transition(
            trigger="day_num_cond", source="month_short_text", dest="day_num"
        )
        self.machine.add_transition(
            trigger="year_not_day_cond", source="month_short_text", dest="end"
        )
        self.machine.add_transition(
            trigger="period_cond", source="month_short_text", dest="period"
        )
        # Period
        self.machine.add_transition(
            trigger="day_num_cond", source="period", dest="day_num"
        )
        self.machine.add_transition(
            trigger="year_not_day_cond", source="period", dest="end"
        )
        # Day number
        self.machine.add_transition(
            trigger="year_num_cond", source="day_num", dest="end"
        )
        self.machine.add_transition(
            trigger="comma_cond", source="day_num", dest="comma"
        )
        self.machine.add_transition(
            trigger="not_comma_not_year_cond",
            source="day_num",
            dest="end",  # dest="nt_end"
        )
        # Comma
        self.machine.add_transition(trigger="year_num_cond", source="comma", dest="end")

    def process_token(self, token, token_idx):
        """

        Args:
          token: 
          token_idx: 

        Returns:

        """
        if self.state == "start":
            if token in full_months:
                self.month_full_cond()
                self.stack.append(token_idx)
            elif token in short_months:
                self.month_short_cond()
                self.stack.append(token_idx)
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "month_full_text":
            # not a day
            if possible_year_not_day(token):
                self.stack.append(token_idx)
                self.year_not_day_cond()
            elif possible_day(token):
                self.day_num_cond()
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "month_short_text":
            if possible_year_not_day(token):
                self.stack.append(token_idx)
                self.year_not_day_cond()
            elif possible_day(token):
                self.day_num_cond()
            elif token == ".":
                self.period_cond()
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "period":
            if possible_year_not_day(token):
                self.stack.append(token_idx)
                self.year_not_day_cond()
            elif possible_day(token):
                self.day_num_cond()
            else:
                self.stack.clear()
                self.reset()
        elif self.state == "day_num":
            if possible_year(token):
                self.stack.append(token_idx)
                self.year_num_cond()
            elif token == ",":
                self.comma_cond()
            else:
                self.stack.append(token_idx - 1)
                self.not_comma_not_year_cond()

        elif self.state == "comma":
            if possible_year(token):
                self.stack.append(token_idx)
                self.year_num_cond()
            else:
                self.stack.clear()
                self.reset()
        if self.state == "end":
            if len(self.stack) == 2:
                first, second = self.stack
                self.indices.add((first, second))
                self.stack.clear()
                self.reset()
            else:
                print(f"Error! In end state and stack contents : {self.stack}")
        if self.state == "nt_end":
            self.reset()
            self.stack.clear()

    def process_tokens(self, token_ls):
        """

        Args:
          token_ls: 

        Returns:

        """
        for idx, token in enumerate(token_ls):
            self.process_token(token.strip().lower(), idx)
        return self.indices


def get_dates(s):
    """

    Args:
      s: 

    Returns:

    """
    text_date_detector = TextDateDetector()
    numeric_date_detector = NumericDateDetector()
    s_toks = ctakes_tok(s)

    def parse_n(n):
        """

        Args:
          n: 

        Returns:

        """
        return parse_numerics(s_toks, n)

    text_date_inds = text_date_detector.process_tokens(s_toks)
    numeric_date_inds = chain.from_iterable(
        map(parse_n, numeric_date_detector.process_tokens(s_toks))
    )
    return sorted([*text_date_inds, *numeric_date_inds])
