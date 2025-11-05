import argparse
import os
import re
from pathlib import Path


PARAMETERS_MINIMAL = (
    'Parameters {\n'
    '    ContentType "HxSpatialGraph"\n'
    '}\n'
)

LABELS_TARGET = (
    'VERTEX { float[3] VertexCoordinates } @1\n'
    'EDGE { int[2] EdgeConnectivity } @2\n'
    'EDGE { int NumEdgePoints } @3\n'
    'POINT { float[3] EdgePointCoordinates } @4\n'
    'POINT { float thickness } @5\n'
)


def _replace_parameters_block(text: str) -> str:
    """
    Replace the first 'Parameters { ... }' block with the minimal block, preserving surrounding whitespace.
    """
    pattern = re.compile(r'Parameters\s*\{.*?\}\s*', re.DOTALL)
    # Only replace the first occurrence (header block)
    new_text, n = pattern.subn(PARAMETERS_MINIMAL + "\n", text, count=1)
    if n == 0:
        raise ValueError("Couldn't find a 'Parameters { ... }' block.")
    return new_text


def _replace_labels_block(text: str) -> str:
    """
    Replace the header labels between the end of Parameters block and the first data marker '@1'
    with the 5 target lines. This avoids touching data sections.
    """
    # Find the end of the first Parameters block we just wrote
    params_end = text.find(PARAMETERS_MINIMAL)
    if params_end == -1:
        # Fallback: locate end of any Parameters block
        m = re.search(r'(Parameters\s*\{.*?\}\s*)', text, re.DOTALL)
        if not m:
            raise ValueError("Couldn't re-locate Parameters block for labels insertion.")
        params_end = m.end()
    else:
        params_end += len(PARAMETERS_MINIMAL)

    # Find the first '@1' data marker (as a whole line)
    m_at1 = re.search(r'(?m)^@1\s*$', text)
    if not m_at1:
        raise ValueError("Couldn't find first '@1' data marker line.")
    at1_start = m_at1.start()

    # Replace anything between end of Parameters block and first '@1' with our labels + a blank line
    before = text[:params_end].rstrip() + "\n\n"
    after = text[at1_start:]  # keep the @1 line and all following
    return before + LABELS_TARGET + "\n" + after


def _delete_data_between_markers(text: str, start_marker: str, end_marker: str) -> str:
    """
    Delete everything from the line that is exactly start_marker (inclusive)
    up to but NOT including the line that is exactly end_marker.
    If start or end isn't found, return text unchanged.
    """
    # Match marker lines exactly, e.g., "^@2\s*$"
    start_re = re.compile(rf'(?m)^@{re.escape(start_marker)}\s*$')
    end_re = re.compile(rf'(?m)^@{re.escape(end_marker)}\s*$')

    start_match = start_re.search(text)
    if not start_match:
        return text  # no start marker, nothing to delete

    end_match = end_re.search(text, start_match.end())
    if not end_match:
        return text  # no following end marker, avoid deleting to EOF

    # Delete from start of start_marker line to start of end_marker line
    return text[:start_match.start()] + text[end_match.start():]


def _delete_identified_graph_data(text: str) -> str:
    """
    Remove identified-graph data blocks:
      1) delete between @2 (inclusive) and @3 (exclusive)
      2) delete between @5 (inclusive) and @6 (exclusive)
    Markers are whole-line '@N' lines, not header labels.
    """
    # The instruction refers to "second occurrence" because '@2' and '@5' appear in labels too.
    # We deliberately match only marker lines (exact '@N' on a line) so we target the data sections.
    text = _delete_data_between_markers(text, "2", "3")
    text = _delete_data_between_markers(text, "5", "6")
    return text


def _renumber_data_markers(text: str) -> str:
    """
    Renumber marker lines only (lines that are exactly '@N'):
      @3 -> @2
      @4 -> @3
      @6 -> @4
      @7 -> @5
    Perform as a single regex with a mapping to avoid cascading collisions.
    """
    mapping = {"3": "2", "4": "3", "6": "4", "7": "5"}

    def repl(m):
        n = m.group(1)
        return f"@{mapping.get(n, n)}"

    # Match only whole-line markers like "@3"
    return re.sub(r'(?m)^@(\d+)\s*$', repl, text)


def reformat_am(read_am_file_path: str, write_am_file_path: str) -> None:
    with open(read_am_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) Replace Parameters block
    text = _replace_parameters_block(text)

    # 2) Replace labels header
    text = _replace_labels_block(text)

    # 3) Remove identified-graph data (@2..@3 and @5..@6)
    text = _delete_identified_graph_data(text)

    # 4) Renumber data markers
    text = _renumber_data_markers(text)

    # Tidy extra blank lines: compress 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    with open(write_am_file_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    read_am_folder = Path(r"C:\...\Spatial Graph Folder (Removed Intermediate Nodes)") # Folder containing spatial graph(s) which have been reconnected, post-processed in Avizo 'Filament' tab to remove intermediate nodes (non-branch or end point nodes) and saved as 'Avizo ascii SpatialGraph (*.am)'
    print(read_am_folder)
    am_files = sorted(read_am_folder.glob("*.am"))
    for src in am_files:
        # write_am_file_path: identical name, remove '.am', add '_reformatted.am'
        dst = src.with_suffix("")  # remove .am
        dst = Path(f"{dst}_reformatted.am")
        reformat_am(str(src), str(dst))
        print(f"OK  {src.name}  ->  {dst.name}")


if __name__ == "__main__":
    main()
