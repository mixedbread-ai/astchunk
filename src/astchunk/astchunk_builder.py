import string
import numpy as np
from typing import Generator

import tree_sitter as ts
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp
import tree_sitter_typescript as tstypescript
import tree_sitter_c as tsc
import tree_sitter_javascript as tsjavascript
import tree_sitter_cpp as tscpp
import tree_sitter_go as tsgo
import tree_sitter_html as tsh
import tree_sitter_ruby as tsruby
import tree_sitter_rust as tsrust
import tree_sitter_php as tphp
import tree_sitter_sql as tsql
import pyrsistent

from astchunk.astnode import ASTNode
from astchunk.astchunk import ASTChunk
from astchunk.preprocessing import ByteRange, preprocess_nws_count, get_nws_count, get_nws_count_direct


class ASTChunkBuilder:
    """
    Attributes:
        - max_chunk_size: Maximum size for each AST chunk, using non-whitespace character count by default.
        - language: Supported languages, currently including python, java, c# and typescript.
        - metadata_template: Type of metadata to store (e.g., start/end line number, path to file, etc).
    """

    def __init__(self, **configs):
        self.max_chunk_size: int = configs["max_chunk_size"]
        self.language: str = configs["language"]
        self.metadata_template: str = configs["metadata_template"]

        if self.language == "python":
            self.parser = ts.Parser(ts.Language(tspython.language()))
        elif self.language == "java":
            self.parser = ts.Parser(ts.Language(tsjava.language()))
        elif self.language == "csharp":
            self.parser = ts.Parser(ts.Language(tscsharp.language()))
        elif self.language == "typescript":
            self.parser = ts.Parser(ts.Language(tstypescript.language_tsx()))
        elif self.language == "c":
            self.parser = ts.Parser(ts.Language(tsc.language()))
        elif self.language == "javascript":
            self.parser = ts.Parser(ts.Language(tsjavascript.language()))
        elif self.language == "cpp":
            self.parser = ts.Parser(ts.Language(tscpp.language()))
        elif self.language == "go":
            self.parser = ts.Parser(ts.Language(tsgo.language()))
        elif self.language == "html":
            self.parser = ts.Parser(ts.Language(tsh.language()))
        elif self.language == "ruby":
            self.parser = ts.Parser(ts.Language(tsruby.language()))
        elif self.language == "rust":
            self.parser = ts.Parser(ts.Language(tsrust.language()))
        elif self.language == "php":
            self.parser = ts.Parser(ts.Language(tphp.language_php()))
        elif self.language == "sql":
            self.parser = ts.Parser(ts.Language(tsql.language()))
        else:
            raise ValueError(f"Unsupported Programming Language: {self.language}!")

    # ------------------------------ #
    #            Step #1             #
    # ------------------------------ #
    def assign_tree_to_windows(
        self, code: str, root_node: ts.Node
    ) -> Generator[list[ASTNode], None, None]:
        """
        Assign AST tree to windows. A window is a tentative chunk consists of ASTNode before being converted into ASTChunk.

        This function serves as a wrapper function for self.assign_nodes_to_windows().
        Additionally, it also
            1. performs preprocessing for efficient AST node size computation.
            2. handles the edge case where the entire AST tree can fit in one window.

        Args:
            code: code to be chunked
            root_node: root node of the AST tree

        Yields:
            Lists (windows) of ASTNode
        """
        # Preprocessing non-whitespace character count
        nws_cumsum = preprocess_nws_count(bytes(code, "utf8"))
        tree_range = ByteRange(root_node.start_byte, root_node.end_byte)
        tree_size = get_nws_count(nws_cumsum, tree_range)

        # If the entire tree can fit in one window, assign tree to window
        if tree_size <= self.max_chunk_size:
            yield [ASTNode(root_node, tree_size)]
        # Otherwise, recursively assign children to windows
        else:
            ancestors = pyrsistent.v(root_node)
            yield from self.assign_nodes_to_windows(
                root_node.children, nws_cumsum, ancestors
            )

    def assign_nodes_to_windows(
        self,
        nodes: list[ts.Node],
        nws_cumsum: np.ndarray,
        ancestors: pyrsistent.pvector,
    ) -> Generator[list[ASTNode], None, None]:
        """
        Assign AST nodes to windows. A window is a tentative chunk consists of ASTNode before being converted into ASTChunk.

        This function:
            1. greedily assigns AST nodes to windows based on their non-whitespace character count.
            2. recursively processes child nodes if the current node exceeds the max chunk size.
            3. keeps track of the ancestors of each node for path construction.

        Args:
            nodes: list of AST nodes to be assigned to windows
            nws_cumsum: cumulative sum of non-whitespace characters
            ancestors: ancestors of the current node

        Yields:
            Lists (windows) of ASTNode
        """
        # Base case: no nodes to assign
        if not nodes:
            yield from []
            return

        # Initialize the current window
        current_window = []
        current_window_size = 0

        for node in nodes:
            node_range = ByteRange(node.start_byte, node.end_byte)
            node_size = get_nws_count(nws_cumsum, node_range)

            # Check if node needs recursive processing (i.e., too large to fit in a window)
            node_exceeds_limit = node_size > self.max_chunk_size

            # Handle the cases where we cannot add the current node to the current window
            # Case 1: current window is empty and node exceeds limit
            # Case 2: current window is not empty and adding the node exceeds limit
            if (len(current_window) == 0 and node_exceeds_limit) or (
                current_window_size + node_size > self.max_chunk_size
            ):
                # Clear current window if not empty
                if len(current_window) > 0:
                    yield current_window
                    current_window = []
                    current_window_size = 0

                # If node still exceeds limit, recursively process the node's children
                if node_exceeds_limit:
                    childs_ancestors = ancestors.append(node)
                    child_windows = list(
                        self.assign_nodes_to_windows(
                            node.children, nws_cumsum, childs_ancestors
                        )
                    )
                    if child_windows:
                        # (optional) Greedily merge adjacent windows from the beginning if merged window does not exceed self.max_chunk_size
                        yield from self.merge_adjacent_windows(child_windows)
                    else:
                        # Leaf node that exceeds the limit — yield as single-node window.
                        # The post-split step in chunkify() will text-split it to fit max_chunk_size.
                        yield [ASTNode(node, node_size, ancestors)]
                else:
                    # Node fits in an empty window
                    current_window.append(ASTNode(node, node_size, ancestors))
                    current_window_size += node_size

            # Case 3: node fits in current window
            else:
                current_window.append(ASTNode(node, node_size, ancestors))
                current_window_size += node_size

        # Add the last window if it's not empty
        if len(current_window) > 0:
            yield current_window

    def merge_adjacent_windows(
        self, ast_windows: list[list[ASTNode]]
    ) -> Generator[list[ASTNode], None, None]:
        """
        Greedily merge adjacent windows of ASTNode if the merged window's total non whitespace character count
        does not exceed max_char_count.

        We choose to merge child windows in this function instead of self.assign_nodes_to_windows() because
        we want to maintain the structure of the original AST as much as possible. Therefore, we should only
        merge windows if all ASTNodes in the window are siblings.

        Args:
            ast_windows: A list of list (windows) of ASTNode

        Yields:
            Lists (windows) of ASTNode with adjacent windows merged where possible
        """
        assert ast_windows, "Expect non-empty ast_windows"

        # Start with a copy of the first list
        merged_windows = [ast_windows[0][:]]

        for window in ast_windows[1:]:
            current_extending_window = merged_windows[-1]

            # Calculate the total character count if we merge
            merged_window_size = sum(n.size for n in current_extending_window) + sum(
                n.size for n in window
            )

            # If merging won't exceed the limit, merge the lists
            if merged_window_size <= self.max_chunk_size:
                current_extending_window.extend(window)
            else:
                # Otherwise, add the current list as a new entry
                merged_windows.append(window[:])

        yield from merged_windows

    # ------------------------------ #
    #            Step #2             #
    # ------------------------------ #
    def add_window_overlapping(
        self, ast_windows: list[list[ASTNode]], chunk_overlap: int
    ) -> list[list[ASTNode]]:
        """
        Extend each window by adding overlapping ASTNodes from the previous and next window.

        Similar to regular document chunking, we add overlapping ASTNodes from the previous and next window
        to each window to provide context. However, we make this step optional since (1) AST Chunking naturally
        avoids breaking the struture of code, hence overlapping is less necessary for maintaining the completeness of
        code blocks (though the additional context may still be useful for downstream tasks); (2) overlapping
        ASTNodes from adjacent windows may cause high variance in chunk size, which makes it difficult to
        control each chunk's token count (especially when the downstream model has a strict limit on context length).

        Args:
            ast_windows: A list of list (windows) of ASTNode
            chunk_overlap: Number of ASTNodes to overlap between adjacent windows

        Returns:
            A list of list (windows) of ASTNode with overlapping ASTNodes added
        """
        assert chunk_overlap >= 0, (
            f"Expect non-negative chunk_overlap, got {chunk_overlap}"
        )

        if chunk_overlap == 0:
            return ast_windows

        new_code_windows = list[list[ASTNode]]()

        for i in range(len(ast_windows)):
            # Create a copy of the current window
            current_node_list = ast_windows[i].copy()
            current_window_size = sum(n.size for n in current_node_list)

            # If there is a previous window, prepend its last chunk_overlap elements (respecting max_chunk_size)
            if i > 0:
                assert len(ast_windows[i - 1]) > 0, (
                    f"Attempting to take elements from an empty window at {i - 1}!"
                )
                prev_window = ast_windows[i - 1]
                candidates = prev_window[-min(chunk_overlap, len(prev_window)) :]
                # Add nodes from the end of candidates (closest to current window) first
                prepend_nodes = []
                for node in reversed(candidates):
                    if current_window_size + node.size <= self.max_chunk_size:
                        prepend_nodes.insert(0, node)
                        current_window_size += node.size
                    else:
                        break
                current_node_list = prepend_nodes + current_node_list

            # If there is a next window, append its first chunk_overlap elements (respecting max_chunk_size)
            if i < len(ast_windows) - 1:
                assert len(ast_windows[i + 1]) > 0, (
                    f"Attempting to take elements from an empty window at {i + 1}!"
                )
                next_window = ast_windows[i + 1]
                candidates = next_window[: min(chunk_overlap, len(next_window))]
                for node in candidates:
                    if current_window_size + node.size <= self.max_chunk_size:
                        current_node_list.append(node)
                        current_window_size += node.size
                    else:
                        break

            new_code_windows.append(current_node_list)

        return new_code_windows

    # ------------------------------ #
    #            Step #3             #
    # ------------------------------ #
    def convert_windows_to_chunks(
        self,
        ast_windows: list[list[ASTNode]],
        repo_level_metadata: dict,
        chunk_expansion: bool,
    ) -> list[ASTChunk]:
        """
        Convert each tentative window of ASTNode into an ASTChunk object.

        This function finalizes the boundary of each chunk and build metadata for each chunk.
        Additionally, it also applies chunk expansion if specified. Chunk expansion is the process of
        adding chunk metadata (e.g., file path, class path) to the beginning of each chunk. It can consist of information
        (1) available in all chunking frameworks (e.g., file path, start line, end line, etc.) and
        (2) specific to AST Chunking (e.g., class path, function path, etc.).
        We found that chunk expansion can be helpful for downstream retrieval and sometimes generation tasks.
        However, it is also worth noting that chunk expansion consumes additional tokens, thereby reducing the number of chunks that can fit in the context window.
        Hence, we make chunk expansion an optional step that can be turned on / off via the `chunk_expansion` flag.

        Args:
            ast_windows: A list of list (windows) of ASTNode
            repo_level_metadata: Repository-level metadata (e.g., repo name, file path)
            chunk_expansion: Whether to perform chunk expansion (i.e., add metadata headers to chunks)

        Returns:
            A list of ASTChunk objects
        """
        ast_chunks = list[ASTChunk]()

        for current_window in ast_windows:
            current_chunk = ASTChunk(
                ast_window=current_window,
                max_chunk_size=self.max_chunk_size,
                language=self.language,
                metadata_template=self.metadata_template,
            )
            current_chunk.build_metadata(repo_level_metadata)

            # (optional) apply chunk expansion
            if chunk_expansion:
                current_chunk.apply_chunk_expansion()
            ast_chunks.append(current_chunk)

        return ast_chunks

    # ------------------------------ #
    #            Step #4             #
    # ------------------------------ #
    def convert_chunks_to_code_windows(self, ast_chunks: list[ASTChunk]) -> list[dict]:
        """
        Convert each ASTChunk object into a code window for downstream integration.

        Args:
            ast_chunks: A list of ASTChunk objects

        Returns:
            A list of code windows, where each code window is a dict with keys "content" and "metadata"
        """
        code_windows = []

        for current_chunk in ast_chunks:
            code_windows.append(current_chunk.to_code_window())

        return code_windows

    # ------------------------------ #
    #            Step #5             #
    # ------------------------------ #
    def post_split_oversized_windows(self, code_windows: list[dict]) -> list[dict]:
        """
        Split any code windows that still exceed max_chunk_size using text-based splitting.

        This acts as a final safety net for cases where a single AST leaf node (e.g., a large
        string literal in minified/bundled code) exceeds the limit and cannot be split further
        via AST decomposition. The text is split at line boundaries when possible, falling back
        to character-level splitting for single lines that exceed the limit.

        Args:
            code_windows: A list of code windows (dicts with "content"/"text" and optional "metadata")

        Returns:
            A list of code windows where every window respects max_chunk_size
        """
        result = []
        for window in code_windows:
            content_key = "text" if "text" in window else "content"
            content = window[content_key]

            if get_nws_count_direct(content) <= self.max_chunk_size:
                result.append(window)
                continue

            metadata = window.get("metadata", {})
            start_line_no = metadata.get("start_line_no", 0)

            # For swebench-lite format, the start line is embedded in _id (e.g. "instance_id_50-75")
            if not metadata and "_id" in window:
                try:
                    line_range = window["_id"].rsplit("_", 1)[1]
                    start_line_no = int(line_range.split("-")[0])
                except (IndexError, ValueError):
                    pass

            base_id = None
            if "_id" in window:
                base_id = window["_id"].rsplit("_", 1)[0]

            sub_parts = self._split_text_to_fit(content)
            for piece_idx, (sub_text, line_offset, line_count) in enumerate(sub_parts):
                sub_start = start_line_no + line_offset
                sub_end = sub_start + line_count - 1
                sub_window = {**window}
                sub_window[content_key] = sub_text

                if "metadata" in window:
                    sub_window["metadata"] = {
                        **metadata,
                        "start_line_no": sub_start,
                        "end_line_no": sub_end,
                        "line_count": line_count,
                        "chunk_size": get_nws_count_direct(sub_text),
                    }

                if base_id is not None:
                    sub_window["_id"] = f"{base_id}_{sub_start}-{sub_end}-{piece_idx}"

                result.append(sub_window)

        return result

    def _split_text_to_fit(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text into pieces where each has <= max_chunk_size non-whitespace characters.

        Splits at line boundaries when possible, falling back to character-level
        splitting for single lines that exceed the limit (e.g., minified code).

        Args:
            text: The text to split

        Returns:
            A list of (sub_text, line_offset, line_count) tuples where line_offset
            is relative to the start of the input text
        """
        lines = text.split("\n")
        result = []
        current_lines = []
        current_nws = 0
        current_line_offset = 0

        for line_idx, line in enumerate(lines):
            line_nws = get_nws_count_direct(line)

            if line_nws > self.max_chunk_size:
                # Flush accumulated lines
                if current_lines:
                    result.append(("\n".join(current_lines), current_line_offset, len(current_lines)))
                    current_lines = []
                    current_nws = 0
                # Character-level split — all pieces share same line number
                for piece in self._split_long_line(line):
                    result.append((piece, line_idx, 1))
                current_line_offset = line_idx + 1
            elif current_nws + line_nws > self.max_chunk_size:
                # Flush current lines and start a new group
                result.append(("\n".join(current_lines), current_line_offset, len(current_lines)))
                current_lines = [line]
                current_nws = line_nws
                current_line_offset = line_idx
            else:
                if not current_lines:
                    current_line_offset = line_idx
                current_lines.append(line)
                current_nws += line_nws

        if current_lines:
            result.append(("\n".join(current_lines), current_line_offset, len(current_lines)))

        return result

    def _split_long_line(self, line: str) -> list[str]:
        """
        Split a single line that exceeds max_chunk_size into pieces by non-whitespace character count.

        Args:
            line: A single line of text whose non-whitespace character count exceeds max_chunk_size

        Returns:
            A list of sub-strings, each with at most max_chunk_size non-whitespace characters
        """
        pieces = []
        current = []
        nws_count = 0

        _whitespace = string.whitespace
        for char in line:
            current.append(char)
            if char not in _whitespace:
                nws_count += 1
                if nws_count >= self.max_chunk_size:
                    pieces.append("".join(current))
                    current = []
                    nws_count = 0

        if current:
            pieces.append("".join(current))

        return pieces

    # ------------------------------ #
    #       AST Chunking Logic       #
    # ------------------------------ #
    def chunkify(self, code: str, **configs) -> list[dict]:
        """
        Parse a piece of code into structual-aware chunks using AST.

        Args:
            code: code to be chunked
            **configs: additional arguments for building chunks and/or chunk metadata
        """
        # step 1: greedily assign AST tree / AST nodes to windows
        #         see self.assign_tree_to_windows() and self.assign_nodes_to_windows() for details
        ast = self.parser.parse(bytes(code, "utf8"))
        ast_windows = list(
            self.assign_tree_to_windows(code=code, root_node=ast.root_node)
        )
        # [after this step]: list[list[ASTNode]] where each sublist represents an AST window

        # step 2 (optional): add overlapping
        #                    for each window, take the last k ASTNodes from the previous window and the first k ASTNodes from the next window
        ast_windows = self.add_window_overlapping(
            ast_windows=ast_windows, chunk_overlap=configs.get("chunk_overlap", 0)
        )
        # [after this step]: list[list[ASTNode]] where each sublist represents an AST window

        # step 3: convert each AST window into an ASTChunk object
        ast_chunks = self.convert_windows_to_chunks(
            ast_windows=ast_windows,
            repo_level_metadata=configs.get("repo_level_metadata", {}),
            chunk_expansion=configs.get("chunk_expansion", False),
        )
        # [after this step]: list[ASTChunk]

        # step 4: convert each ASTChunk to a code window for downstream integration
        code_windows = self.convert_chunks_to_code_windows(ast_chunks=ast_chunks)
        # [after this step]: list[dict] where each dict represents a code window

        # step 5: post-split any oversized windows using text-based splitting
        #         this handles cases where a single AST leaf node exceeds max_chunk_size
        code_windows = self.post_split_oversized_windows(code_windows)
        # [after this step]: list[dict] where each dict respects max_chunk_size

        return code_windows
