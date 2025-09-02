#!/usr/bin/env python3
import sys
import argparse
import re


def _generate_single_tree(text_block: str, compactness: int) -> str:
    """
    Converts a single, contiguous block of indented text into a tree
    with a specified level of compactness.
    """
    # Define drawing sets for each compactness level.
    # Format: (BRANCH, LAST_BRANCH, VERTICAL, INDENT)
    # drawing_sets = [
    #     ("├ ", "└ ", "│ ", "  "),  # compactness = 0
    #     ("├─ ", "└─ ", "│  ", "   "),  # compactness = 1
    #     ("├── ", "└── ", "│   ", "    "),  # compactness = 2 (default)
    #     ("├─── ", "└─── ", "│    ", "     "),  # compactness = 3
    # ]
    # try:
    #     BRANCH, LAST_BRANCH, VERTICAL, INDENT = drawing_sets[compactness]
    # except IndexError:
    #     raise ValueError("Compactness level must be between 0 and 3.")
    BRANCH = "├" + "─" * compactness + " "
    LAST_BRANCH = "└" + "─" * compactness + " "
    VERTICAL = "│" + " " * (compactness + 1)
    INDENT = " " * (compactness + 2)

    lines = text_block.strip().split('\n')
    if not lines:
        return ""

    raw_nodes = []
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        content = line.strip()
        raw_nodes.append((indent, content))

    non_zero_indents = sorted(list(set(indent for indent, _ in raw_nodes if indent > 0)))
    indent_unit = non_zero_indents[0] if non_zero_indents else 2

    nodes = [(indent // indent_unit, content) for indent, content in raw_nodes]

    output_lines = []
    is_last_at_depth = {}

    for i, (depth, content) in enumerate(nodes):
        if depth == 0:
            output_lines.append(content)
            is_last_at_depth = {}
            continue

        is_last = True
        for next_depth, _ in nodes[i + 1 :]:
            if next_depth == depth:
                is_last = False
                break
            if next_depth < depth:
                break

        is_last_at_depth[depth] = is_last

        prefix = ""
        for d in range(depth):
            if d == depth - 1:
                prefix += LAST_BRANCH if is_last else BRANCH
            else:
                if is_last_at_depth.get(d + 1, False):
                    prefix += INDENT
                else:
                    prefix += VERTICAL

        output_lines.append(prefix + content)

    return "\n".join(output_lines)


def generate_tree_from_text(text_content: str, compactness: int = 2) -> str:
    """
    Converts indented text into a text-based tree structure,
    preserving blank lines between separate trees and allowing for
    different compactness levels.

    Args:
        text_content: A string containing the indented text.
        compactness: An integer (0-3) controlling the visual spacing of the tree.
                     0 is most compact, 3 is most spread out. Defaults to 2.

    Returns:
        A string formatted as a text tree.
    """
    tree_blocks = re.split(r'\n\s*\n', text_content)

    processed_trees = []
    for block in tree_blocks:
        if block.strip():
            processed_trees.append(_generate_single_tree(block, compactness))

    return "\n\n".join(processed_trees)


def main():
    """
    Main function to run the script from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Convert an indented text file into a text tree representation. Preserves blank lines between trees.",
        epilog="If no file is provided, input is read from stdin.",
    )
    parser.add_argument(
        "infile",
        nargs="?",
        type=argparse.FileType('r'),
        default=sys.stdin,
        help="The input file to process. Reads from stdin if not specified.",
    )
    parser.add_argument(
        "-c",
        "--compactness",
        type=int,
        default=2,
        help="Set the tree compactness level. 0 is most compact. Default: 2.",
    )

    args = parser.parse_args()

    with args.infile as f:
        input_text = f.read()

    tree_output = generate_tree_from_text(input_text, compactness=args.compactness)

    print(tree_output)


if __name__ == "__main__":
    main()
