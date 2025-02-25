from typing import Optional, Dict

import numpy as np

from transformers.utils import ExplicitEnum, add_end_docstrings
from transformers.pipelines.base import (
    PIPELINE_INIT_ARGS,
    GenericTensor,
    Pipeline,
)
from transformers.data.processors.utils import DataProcessor

from .__init__ import ctakes_tok


def sigmoid(_outputs):
    """

    Args:
      _outputs:

    Returns:

    """
    return 1.0 / (1.0 + np.exp(-_outputs))


def softmax(_outputs):
    """

    Args:
      _outputs:

    Returns:

    """
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class ClassificationFunction(ExplicitEnum):
    """ """

    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (`bool`, *optional*, defaults to `False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    """,
)
class ClassificationPipeline(Pipeline):
    """Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification
    examples](../task_summary#sequence-classification) for more information.

    This text classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).

    If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax
    over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text-classification).

    Args:

    Returns:

    """

    return_all_scores = False
    function_to_apply = ClassificationFunction.NONE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _sanitize_parameters(
        self,
        return_all_scores=None,
        function_to_apply=None,
        task_processor: Optional[DataProcessor] = None,
        **tokenizer_kwargs,
    ):
        """

        Args:
          return_all_scores:  (Default value = None)
          function_to_apply:  (Default value = None)
          task_processor: Optional[DataProcessor]:  (Default value = None)
          **tokenizer_kwargs:

        Returns:

        """
        preprocess_params = tokenizer_kwargs

        postprocess_params = {}
        if (
            hasattr(self.model.config, "return_all_scores")
            and return_all_scores is None
        ):
            return_all_scores = self.model.config.return_all_scores

        if return_all_scores is not None:
            postprocess_params["return_all_scores"] = return_all_scores

        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction[function_to_apply.upper()]

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply

        if task_processor is not None:
            postprocess_params["task_processor"] = task_processor
        elif "task_processor" not in self._postprocess_params:
            raise ValueError("Task_processor was never initialized")

        return preprocess_params, {}, postprocess_params

    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                One or several texts (or one list of prompts) to classify.
            return_all_scores (`bool`, *optional*, defaults to `False`):
                Whether to return scores for all labels.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function on the output.
                - If the model has several labels, will apply the softmax function on the output.

                Possible values are:

                - `"sigmoid"`: Applies the sigmoid function on the output.
                - `"softmax"`: Applies the softmax function on the output.
                - `"none"`: Does not apply any function on the output.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.

            If `self.return_all_scores=True`, one such dictionary is returned per label.
        """
        result = super().__call__(*args, **kwargs)
        if isinstance(args[0], str) and isinstance(result, dict):
            # This pipeline is odd, and return a list when single item is run
            return [result]
        else:
            return result

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        """

        Args:
          inputs:
          **tokenizer_kwargs:

        Returns:

        """
        return_tensors = self.framework
        return self.tokenizer(
            ctakes_tok(inputs),
            # not sure how to pass this one upstream in a non-messy way
            max_length=512,  # self.tokenizer.model_max_length,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )

    def _forward(self, model_inputs):
        """

        Args:
          model_inputs:

        Returns:

        """
        return self.model(**model_inputs)

    def postprocess(
        self,
        model_outputs,
        # function_to_apply=None,
        task_processor: DataProcessor,
        return_all_scores=False,
    ):
        """

        Args:
          model_outputs:
          # function_to_apply:  (Default value = None)
          task_processor: DataProcessor:
          return_all_scores:  (Default value = False)

        Returns:

        """
        # Using task processor labels here
        # instead of id2label and label2id from the config
        # for the same reasons described in tagging.py
        label_list = task_processor.get_labels()

        if len(label_list) == 1:
            function_to_apply = ClassificationFunction.SIGMOID
        elif len(label_list) > 1:
            function_to_apply = ClassificationFunction.SOFTMAX
        else:
            function_to_apply = ClassificationFunction.NONE

        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()
        
        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            raise ValueError(
                f"Unrecognized `function_to_apply` argument: {function_to_apply}"
            )

        if return_all_scores:
            # By default scores are in a nested list that
            # throws off item() but not max() and argmax()
            # thus this guard
            if len(scores.shape) > 1:
                scores = scores[0]
            return [
                {"label": label_list[i], "score": score.item()}
                for i, score in enumerate(scores)
            ]
        else:
            return {
                "label": label_list[scores.argmax().item()],
                "score": scores.max().item(),
            }
