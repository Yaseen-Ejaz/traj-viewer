"""
Classify a single test-related file diff from a `unidiff.PatchSet`.

This function takes one `unidiff.patch.PatchedFile` object and applies the
same deterministic classification logic used by the original RQ2 labeler.
It analyzes the added and deleted lines in the patch file, extracts the same
signals used by the original script, and returns the resulting classification.

The function does not require the original CSV, parquet, or repository-specific
file-loading logic. It expects that the caller has already parsed a diff using
`unidiff.PatchSet` and is passing in one patched file at a time.

Parameters
----------
patch_file : unidiff.patch.PatchedFile
    A single patched file object from a `unidiff.PatchSet`.

    Example:
        from unidiff import PatchSet

        patch_set = PatchSet.from_filename("patch.diff")

        for patch_file in patch_set:
            result = classify_patch_file(patch_file)
            print(result)

Returns
-------
dict[str, object]
    A dictionary containing the classification result and supporting signals.

    Returned keys:
        file_path : str
            The normalized path of the patched file.

        deterministic_label : str
            The final deterministic label. This is empty when the result is
            flagged as requiring manual review.

        suggested_label : str
            The label suggested by the deterministic decision tree.

        confidence : str
            The confidence assigned by the original classification logic.
            One of: "high", "medium", or "low".

        needs_review : bool
            Whether the classification should be manually reviewed.

        review_reason : str
            Explanation for why the label was selected or why review is needed.

        signals : dict[str, object]
            The extracted deterministic signals used by the classifier, such as
            test additions, test deletions, assertion counts, expected-output
            line counts, standalone-file share, support-file share, and deletion
            share.

Classification Labels
---------------------
The possible suggested labels are:

    - "Coverage removal/replacement"
    - "Additive test coverage"
    - "Expected-output adaptation"
    - "Assertion weakening/narrowing"
    - "Test-support/scaffolding change"
    - "Standalone reproduction test creation"

Notes
-----
This function preserves the original classification and decision-tree logic.
Only the input mechanism has changed: instead of reading rows from CSV/parquet
files and then locating patch files on disk, this function derives the required
signals directly from the provided `unidiff` patched file object.
"""


def classify_patch_test_file(patch_file):
    """
    Classify a single unidiff PatchedFile using the same deterministic
    classification logic as the original RQ2 labeler.

    Parameters
    ----------
    patch_file:
        A single file object from a `unidiff.PatchSet`.

        Example:
            from unidiff import PatchSet

            patch_set = PatchSet.from_filename("patch.diff")

            for patch_file in patch_set:
                result = classify_patch_file(patch_file)
                print(result)

    Returns
    -------
    dict[str, object]
        A dictionary containing:
            - deterministic_label
            - suggested_label
            - confidence
            - needs_review
            - review_reason
            - signals

    Notes
    -----
    This function preserves the original classification logic. The only
    adaptation is that test additions, deletions, changed lines, file path,
    standalone/support status, and deleted-file status are derived directly
    from the provided unidiff PatchedFile object.
    """
    import re
    from pathlib import PurePosixPath

    LABEL_COVERAGE = "Coverage removal/replacement"
    LABEL_ADDITIVE = "Additive test coverage"
    LABEL_EXPECTED = "Expected-output adaptation"
    LABEL_WEAKENED = "Assertion weakening/narrowing"
    LABEL_SCAFFOLDING = "Test-support/scaffolding change"
    LABEL_STANDALONE = "Standalone reproduction test creation"

    ASSERTION_PAT = re.compile(
        r"\b(assert|self\.assert\w+|pytest\.raises|raises\(|warns\(|match=|expected|"
        r"assertEqual|assertIn|assertNotIn|assertRegex|assertWarns)\b"
    )
    TEST_DEF_PAT = re.compile(r"^\s*(def\s+test_|class\s+Test|class\s+\w+Test\b)")
    WEAK_PARTIAL_PAT = re.compile(
        r"\b(in\s+|not\s+in\s+|contains|startswith|endswith|len\(|isinstance\(|"
        r"any\(|all\(|count\(|find\(|search\(|match\(|re\.|if\s+|continue|skip|"
        r"pytest\.skip|assertIn|assertNotIn|assertRegex)\b"
    )
    CONCRETE_EXPECTED_PAT = re.compile(
        r"([\"'][^\"']{2,}[\"']|\bexpected\b|\bwarning\b|\berror\b|\brepr\b|"
        r"\bmessage\b|\bpath\b|\boutput\b|\bresult\b|\bmatch=)"
    )

    SUPPORT_FILENAMES = {
        "__init__.py",
        "admin.py",
        "apps.py",
        "conftest.py",
        "fixtures.py",
        "models.py",
        "settings.py",
        "urls.py",
        "wsgi.py",
    }

    STANDALONE_EXACT = {
        "simple_test.py",
        "test_fix.py",
        "test_issue.py",
        "test_poly.py",
        "test_edge_cases.py",
        "test_pr_fix.py",
        "test_radd_fix.py",
    }

    def normalize_diff_path(path):
        path = str(path or "")
        if path in {"", "/dev/null"}:
            return ""
        if path.startswith("a/") or path.startswith("b/"):
            return path[2:]
        return path

    def get_patch_file_path(patch_file):
        target = normalize_diff_path(getattr(patch_file, "target_file", ""))
        source = normalize_diff_path(getattr(patch_file, "source_file", ""))

        if target and target != "/dev/null":
            return target
        return source

    def is_standalone_path(path):
        parts = PurePosixPath(path).parts
        if not parts:
            return False

        filename = parts[-1]

        if parts[0] in {"tmp", "temp"}:
            return True

        if "test_project" in parts and parts[0] != "tests":
            return True

        if len(parts) == 1:
            return (
                filename in STANDALONE_EXACT
                or filename.startswith(("reproduce", "repro_"))
                or filename.startswith("test_")
                or filename.endswith("_test.py")
            )

        return False

    def is_support_path(path):
        filename = PurePosixPath(path).name

        if filename in SUPPORT_FILENAMES:
            return True

        return any(
            part in {"fixtures", "fixture", "support", "helpers"}
            for part in PurePosixPath(path).parts
        )

    def count_matching(lines, pattern):
        return sum(1 for line in lines if pattern.search(line))

    added_lines = []
    deleted_lines = []

    for hunk in patch_file:
        for line in hunk:
            if getattr(line, "is_added", False):
                added_lines.append(str(getattr(line, "value", "")))
            elif getattr(line, "is_removed", False):
                deleted_lines.append(str(getattr(line, "value", "")))

    additions = len(added_lines)
    deletions = len(deleted_lines)
    churn = max(additions + deletions, additions + deletions)
    deletion_share = deletions / churn if churn else 0.0

    file_path = get_patch_file_path(patch_file)

    standalone_churn = churn if is_standalone_path(file_path) else 0
    support_churn = churn if is_support_path(file_path) else 0

    deleted_file_count = 1 if getattr(patch_file, "is_removed_file", False) else 0

    added_assertions = count_matching(added_lines, ASSERTION_PAT)
    deleted_assertions = count_matching(deleted_lines, ASSERTION_PAT)
    added_test_defs = count_matching(added_lines, TEST_DEF_PAT)
    deleted_test_defs = count_matching(deleted_lines, TEST_DEF_PAT)
    added_partial = count_matching(added_lines, WEAK_PARTIAL_PAT)
    deleted_expected = count_matching(deleted_lines, CONCRETE_EXPECTED_PAT)
    added_expected = count_matching(added_lines, CONCRETE_EXPECTED_PAT)

    signals = {
        "test_additions": additions,
        "test_deletions": deletions,
        "test_churn": churn,
        "deletion_share": deletion_share,
        "file_count": 1 if file_path else 0,
        "standalone_churn": standalone_churn,
        "standalone_share": standalone_churn / churn if churn else 0.0,
        "support_churn": support_churn,
        "support_share": support_churn / churn if churn else 0.0,
        "deleted_file_count": deleted_file_count,
        "added_assertions": added_assertions,
        "deleted_assertions": deleted_assertions,
        "added_test_defs": added_test_defs,
        "deleted_test_defs": deleted_test_defs,
        "added_partial_assertions": added_partial,
        "deleted_expected_lines": deleted_expected,
        "added_expected_lines": added_expected,
        "patch_found": True,
    }

    additions = int(signals["test_additions"])
    deletions = int(signals["test_deletions"])
    churn = int(signals["test_churn"])
    deletion_share = float(signals["deletion_share"])
    standalone_share = float(signals["standalone_share"])
    support_share = float(signals["support_share"])
    deleted_file_count = int(signals["deleted_file_count"])
    added_assertions = int(signals["added_assertions"])
    deleted_assertions = int(signals["deleted_assertions"])
    added_test_defs = int(signals["added_test_defs"])
    deleted_test_defs = int(signals["deleted_test_defs"])
    added_partial = int(signals["added_partial_assertions"])
    deleted_expected = int(signals["deleted_expected_lines"])
    added_expected = int(signals["added_expected_lines"])

    coverage_decisive = (
        deleted_file_count > 0 or deletions >= 100 or deleted_test_defs >= 2
    )
    coverage_candidate = deletions >= 5 and deletion_share >= 0.20
    expected_signal = bool(deleted_expected and added_expected)
    weakening_signal = bool(
        added_partial >= 2 and (deleted_assertions >= 1 or deleted_expected >= 1)
    )
    additive_signal = bool(added_test_defs or added_assertions > deleted_assertions)

    if coverage_decisive:
        suggested, confidence, needs_review, reason = (
            LABEL_COVERAGE,
            "high",
            False,
            "decisive coverage-removal signal: deleted file, large deletion, deletion-heavy churn, or several deleted test units",
        )

    elif standalone_share >= 0.60 and additions >= 20 and not coverage_candidate:
        suggested, confidence, needs_review, reason = (
            LABEL_STANDALONE,
            "high",
            False,
            "standalone/root-level reproduction artifacts dominate test churn",
        )

    elif coverage_candidate:
        if support_share >= 0.60 and added_test_defs == 0 and added_assertions <= 1:
            suggested, confidence, needs_review, reason = (
                LABEL_SCAFFOLDING,
                "medium",
                True,
                "deletion threshold fired inside test-support files; review scaffolding versus coverage removal",
            )
        elif weakening_signal and not expected_signal:
            suggested, confidence, needs_review, reason = (
                LABEL_WEAKENED,
                "medium",
                True,
                "assertion-weakening signal overlaps with calibrated removal/replacement flag",
            )
        elif expected_signal:
            suggested, confidence, needs_review, reason = (
                LABEL_EXPECTED,
                "medium",
                True,
                "expected-output signal overlaps with calibrated removal/replacement flag",
            )
        else:
            suggested, confidence, needs_review, reason = (
                LABEL_COVERAGE,
                "medium",
                True,
                "calibrated removal/replacement flag fired, but semantic review is needed",
            )

    elif weakening_signal and expected_signal and added_partial < 4:
        suggested, confidence, needs_review, reason = (
            LABEL_EXPECTED,
            "medium",
            True,
            "expected-output and assertion-weakening signals overlap; review oracle specificity",
        )

    elif weakening_signal and not (added_test_defs and additions >= deletions * 3):
        suggested, confidence, needs_review, reason = (
            LABEL_WEAKENED,
            "high",
            False,
            "deleted concrete assertions are replaced by partial, conditional, or presence-style checks",
        )

    elif support_share >= 0.60 and added_test_defs == 0 and added_assertions <= 1:
        suggested, confidence, needs_review, reason = (
            LABEL_SCAFFOLDING,
            "high",
            False,
            "test-support files dominate and there is little direct behavioral assertion churn",
        )

    elif deletions == 0 and standalone_share < 0.50:
        suggested, confidence, needs_review, reason = (
            LABEL_ADDITIVE,
            "high",
            False,
            "test additions with no test deletions inside the normal suite",
        )

    elif additive_signal and additions >= max(10, deletions * 3):
        suggested, confidence, needs_review, reason = (
            LABEL_ADDITIVE,
            "medium",
            bool(expected_signal or weakening_signal),
            "new or extended in-suite tests dominate; review if nearby oracle edits are also present",
        )

    elif deleted_assertions and added_expected and deleted_expected:
        suggested, confidence, needs_review, reason = (
            LABEL_EXPECTED,
            "medium",
            False,
            "existing concrete expected/assertion lines are changed to new concrete expected/assertion lines",
        )

    elif additive_signal:
        suggested, confidence, needs_review, reason = (
            LABEL_ADDITIVE,
            "medium",
            additions < max(5, deletions * 2),
            "new or extended in-suite tests appear to dominate, with only minor or ambiguous deletion",
        )

    elif support_share > 0:
        suggested, confidence, needs_review, reason = (
            LABEL_SCAFFOLDING,
            "medium",
            True,
            "test-support files are involved, but the primary behavior is not decisive",
        )

    elif churn:
        suggested, confidence, needs_review, reason = (
            LABEL_EXPECTED,
            "low",
            True,
            "test edit exists, but deterministic signals are weak; expected-output update is the closest fallback",
        )

    else:
        suggested, confidence, needs_review, reason = (
            LABEL_ADDITIVE,
            "low",
            True,
            "no test churn found for this row",
        )

    return {
        "file_path": file_path,
        "deterministic_label": "" if needs_review else suggested,
        "suggested_label": suggested,
        "confidence": confidence,
        "needs_review": needs_review,
        "review_reason": reason,
        "signals": signals,
    }
