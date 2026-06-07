#!/usr/bin/env python3
"""
Classify files touched by unified diff patches.

This module classifies each file touched by a `.diff` file as one of:

- "documentation"
- "test"
- "code"

The classification precedence is:

1. documentation
2. test
3. code

This means that if a file could match more than one category, it receives the
first matching category. For example, `docs/test_example.md` is classified as
"documentation", not "test".

Requirements:
    pip install unidiff

Basic usage with a diff file:

    from diff_classifier import classify_diff_file

    result = classify_diff_file("example.diff")
    print(result)

Example output:

    {
        "src/parser.py": "code",
        "tests/test_parser.py": "test",
        "README.md": "documentation"
    }

Usage with diff text:

    from diff_classifier import classify_diff_text

    with open("example.diff", encoding="utf-8") as f:
        diff_text = f.read()

    result = classify_diff_text(diff_text)
    print(result)

Usage with a unidiff PatchSet:

    from unidiff import PatchSet
    from diff_classifier import classify_patch_set

    with open("example.diff", encoding="utf-8") as f:
        patch_set = PatchSet(f)

    result = classify_patch_set(patch_set)
    print(result)

Usage with a single unidiff PatchFile/PatchedFile:

    from unidiff import PatchSet
    from diff_classifier import classify_patch_file

    with open("example.diff", encoding="utf-8") as f:
        patch_set = PatchSet(f)

    for patch_file in patch_set:
        classification = classify_patch_file(patch_file)
        print(patch_file.path, classification)

Command-line usage:

    python diff_classifier.py example.diff

Command-line output:

    {
      "src/parser.py": "code",
      "tests/test_parser.py": "test",
      "README.md": "documentation"
    }

Main functions:
    classify_diff_file(diff_file):
        Reads a `.diff` file from disk and returns a dictionary mapping each
        touched file path to its classification.

    classify_diff_text(diff_text):
        Parses unified diff text and returns a dictionary mapping each touched
        file path to its classification.

    classify_patch_set(patch_set):
        Classifies every file in a unidiff PatchSet.

    classify_patch_file(patch_file):
        Classifies one file object from a unidiff PatchSet.

Return format:
    dict[str, str]

    The keys are touched file paths.
    The values are classification labels.

    Example:

        {
            "src/main.py": "code",
            "tests/test_main.py": "test",
            "CHANGELOG.md": "documentation"
        }
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

_DOCUMENTATION_SUFFIXES = {
    ".adoc",
    ".asciidoc",
    ".markdown",
    ".md",
    ".mdown",
    ".mdx",
    ".mkd",
    ".rdoc",
    ".rst",
    ".rtf",
    ".text",
    ".txt",
}

_DOCUMENTATION_FILENAMES = {
    "authors",
    "changelog",
    "contributing",
    "copying",
    "install",
    "license",
    "notice",
    "readme",
    "security",
}

_EXCLUDED_TEST_DIRECTORIES = {
    "doc",
    "docs",
    "documentation",
    "example",
    "examples",
}

_TEST_DIRECTORIES = {
    "test",
    "testing",
    "tests",
}


def clean_patch_file_path(value: str) -> str:
    """
    Normalize a file path extracted from a unified diff header.

    Removes Git's leading a/ and b/ prefixes.
    Returns an empty string for /dev/null.
    """

    path = value.strip().split("\t", maxsplit=1)[0].split(" ", maxsplit=1)[0]

    if not path or path == "/dev/null":
        return ""

    if path.startswith(("a/", "b/")):
        return path[2:]

    return path


def is_documentation_path(path: str) -> bool:
    """
    Return True if the path should be classified as documentation.
    """

    path_obj = PurePosixPath(path)
    file_name = path_obj.name.lower()
    stem = path_obj.stem.lower()
    suffix = path_obj.suffix.lower()

    return (
        suffix in _DOCUMENTATION_SUFFIXES
        or file_name in _DOCUMENTATION_FILENAMES
        or stem in _DOCUMENTATION_FILENAMES
    )


def is_test_path(path: str) -> bool:
    """
    Return True if the path should be classified as test code.
    """

    path_obj = PurePosixPath(path)
    lowered_parts = [part.lower() for part in path_obj.parts]
    file_name = path_obj.name.lower()

    if any(part in _EXCLUDED_TEST_DIRECTORIES for part in lowered_parts):
        return False

    if any(part in _TEST_DIRECTORIES for part in lowered_parts[:-1]):
        return True

    return (
        fnmatch(file_name, "test_*.py")
        or fnmatch(file_name, "*_test.py")
        or file_name in {"conftest.py", "tests.py"}
    )


def classify_path(path: str) -> str:
    """
    Classify one file path as documentation, test, or code.

    Precedence:
    1. documentation
    2. test
    3. code
    """

    if is_documentation_path(path):
        return "documentation"

    if is_test_path(path):
        return "test"

    return "code"


def get_patch_file_path(patch_file: Any) -> str:
    """
    Extract the best available path from a unidiff PatchedFile/PatchFile object.
    """

    source_path = clean_patch_file_path(getattr(patch_file, "source_file", ""))
    target_path = clean_patch_file_path(getattr(patch_file, "target_file", ""))
    direct_path = clean_patch_file_path(getattr(patch_file, "path", ""))

    return direct_path or target_path or source_path


def classify_patch_file(patch_file: Any) -> str:
    """
    Classify a single unidiff PatchedFile/PatchFile object.

    Input:
        One file object from a PatchSet.

    Output:
        One of:
        - documentation
        - test
        - code
    """

    path = get_patch_file_path(patch_file)

    if not path:
        return "code"

    return classify_path(path)


def classify_patch_set(patch_set: Any) -> dict[str, str]:
    """
    Classify every file in a unidiff PatchSet.

    Returns:
        {
            "src/example.py": "code",
            "tests/test_example.py": "test",
            "README.md": "documentation",
        }
    """

    classifications: dict[str, str] = {}

    for patch_file in patch_set:
        path = get_patch_file_path(patch_file)

        if not path:
            continue

        classifications[path] = classify_patch_file(patch_file)

    return classifications


def classify_diff_text(diff_text: str) -> dict[str, str]:
    """
    Parse unified diff text and classify every touched file.

    Requires:
        pip install unidiff
    """

    try:
        from unidiff import PatchSet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "classify_diff_text requires the 'unidiff' package. "
            "Install it with: pip install unidiff"
        ) from exc

    patch_set = PatchSet.from_string(diff_text)
    return classify_patch_set(patch_set)


def classify_diff_file(diff_file: str | Path) -> dict[str, str]:
    """
    Read a .diff file and classify every touched file.

    Example:
        result = classify_diff_file("example.diff")
        print(result)
    """

    diff_text = Path(diff_file).read_text(encoding="utf-8", errors="replace")
    return classify_diff_text(diff_text)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Classify files touched by a unified diff."
    )
    parser.add_argument(
        "diff_file",
        type=Path,
        help="Path to a .diff file.",
    )

    args = parser.parse_args()
    result = classify_diff_file(args.diff_file)

    print(json.dumps(result, indent=2))
