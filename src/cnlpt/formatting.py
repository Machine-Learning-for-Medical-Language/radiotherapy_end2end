from tabulate import tabulate


def tabulate_report(report):
    """

    Args:
      report: Raw classification results report from cnlp_processors

    Returns:
      Tabulated report with information about missing classes
    """
    class_specific_keys = {"f1", "recall", "precision", "support"}
    class_specific_report = {k: v for k, v in report.items() if k in class_specific_keys}
    class_agnostic_report = {"acc": report["acc"]}
    headers = ["Score", *report["total_classes"]]

    report_table = tabulate(
        [[k, *v] for k, v in class_specific_report.items()], headers=headers
    )
    class_agnostic_addendum = "\n".join(
        sorted((f"{k}: {v}" for k, v in class_agnostic_report.items()))
    )

    false_negative = set(report["classes present in labels"]) - set(
        report["classes present in predictions"]
    )
    false_positive = set(report["classes present in predictions"]) - set(
        report["classes present in labels"]
    )

    missing_from_preds = ""
    extra_from_preds = ""
    if false_negative:
        missing_from_preds = f"Gold classes missing in predictions (whole class false negative): {sorted(false_negative)}"

    if false_positive:
        extra_from_preds = f"Prediction classes which do not appear in gold (whole class false positive): {sorted(false_positive)}"
    class_agnostic_addendum = "\n".join(
        [
            *sorted((f"{k}: {v}" for k, v in class_agnostic_report.items())),
            missing_from_preds,
            extra_from_preds,
        ]
    )
    return "\n\n" + report_table + "\n\n" + class_agnostic_addendum + "\n\n"
