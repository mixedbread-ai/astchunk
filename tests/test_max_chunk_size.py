#!/usr/bin/env python3
"""
Tests to verify that max_chunk_size is strictly enforced across all chunking options.
"""

import pytest
from astchunk import ASTChunkBuilder, get_nws_count_direct


def _make_bundled_js(num_statements: int) -> str:
    """Generate JS with an oversized inline string literal (single AST leaf node)."""
    filler = ";".join(f"var v{i}={i}" for i in range(num_statements))
    return (
        '"use strict";\n'
        "function setup() {\n"
        "  console.log('init');\n"
        "}\n"
        "\n"
        "function createWorker() {\n"
        f"  return inlineWorker('{filler}');\n"
        "}\n"
        "\n"
        "function teardown() {\n"
        "  console.log('done');\n"
        "}\n"
    )


PYTHON_CODE = '''
class MyClass:
    def method_one(self):
        x = 1 + 2 + 3 + 4 + 5
        y = 'hello world this is a long string'
        return x + len(y)

    def method_two(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [x * 2 for x in a]
        return sum(b)

    def method_three(self):
        result = {}
        for i in range(100):
            result[i] = i ** 2
        return result

    def method_four(self):
        data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        filtered = {k: v for k, v in data.items() if 'key' in k}
        return filtered
'''


def _assert_all_chunks_within_limit(chunks, max_chunk_size):
    """Helper to assert all chunks respect max_chunk_size."""
    for i, chunk in enumerate(chunks):
        content = chunk.get("content") or chunk.get("text", "")
        size = get_nws_count_direct(content)
        assert size <= max_chunk_size, (
            f"Chunk {i} has size {size} which exceeds max_chunk_size {max_chunk_size}"
        )


@pytest.mark.parametrize("max_chunk_size", [50, 100, 200])
def test_basic_chunking_respects_limit(max_chunk_size):
    """Chunks without overlap or expansion should respect max_chunk_size."""
    builder = ASTChunkBuilder(
        max_chunk_size=max_chunk_size,
        language="python",
        metadata_template="default",
    )
    chunks = builder.chunkify(
        PYTHON_CODE,
        repo_level_metadata={"filepath": "test.py"},
    )
    assert len(chunks) > 0
    _assert_all_chunks_within_limit(chunks, max_chunk_size)


@pytest.mark.parametrize("max_chunk_size", [50, 100, 200])
@pytest.mark.parametrize("chunk_overlap", [1, 2, 5])
def test_overlap_respects_limit(max_chunk_size, chunk_overlap):
    """Chunks with overlap should still respect max_chunk_size."""
    builder = ASTChunkBuilder(
        max_chunk_size=max_chunk_size,
        language="python",
        metadata_template="default",
    )
    chunks = builder.chunkify(
        PYTHON_CODE,
        chunk_overlap=chunk_overlap,
        repo_level_metadata={"filepath": "test.py"},
    )
    assert len(chunks) > 0
    _assert_all_chunks_within_limit(chunks, max_chunk_size)


@pytest.mark.parametrize("max_chunk_size", [50, 100, 200])
def test_chunk_expansion_respects_limit(max_chunk_size):
    """Chunks with expansion enabled should still respect max_chunk_size."""
    builder = ASTChunkBuilder(
        max_chunk_size=max_chunk_size,
        language="python",
        metadata_template="default",
    )
    chunks = builder.chunkify(
        PYTHON_CODE,
        chunk_expansion=True,
        repo_level_metadata={"filepath": "test.py"},
    )
    assert len(chunks) > 0
    _assert_all_chunks_within_limit(chunks, max_chunk_size)


@pytest.mark.parametrize("max_chunk_size", [50, 100, 200])
@pytest.mark.parametrize("chunk_overlap", [1, 2, 5])
def test_overlap_and_expansion_respects_limit(max_chunk_size, chunk_overlap):
    """Chunks with both overlap and expansion should still respect max_chunk_size."""
    builder = ASTChunkBuilder(
        max_chunk_size=max_chunk_size,
        language="python",
        metadata_template="default",
    )
    chunks = builder.chunkify(
        PYTHON_CODE,
        chunk_overlap=chunk_overlap,
        chunk_expansion=True,
        repo_level_metadata={"filepath": "test.py"},
    )
    assert len(chunks) > 0
    _assert_all_chunks_within_limit(chunks, max_chunk_size)


# ------------------------------------------------------------------ #
#  Oversized leaf node tests (e.g., minified / bundled JS)           #
# ------------------------------------------------------------------ #

def test_oversized_leaf_node_is_not_dropped():
    """A large string literal (single AST leaf) must not be silently dropped."""
    content = _make_bundled_js(num_statements=10_000)
    builder = ASTChunkBuilder(max_chunk_size=2048, language="javascript", metadata_template="default")
    chunks = builder.chunkify(content, repo_level_metadata={"filepath": "bundle.js"})

    # Reconstruct all chunk text and verify the filler content is present
    all_text = " ".join(c["content"] for c in chunks)
    assert "var v0=0" in all_text
    assert "var v9999=9999" in all_text


def test_oversized_leaf_node_respects_limit():
    """Post-split of an oversized leaf node must produce chunks within the limit."""
    content = _make_bundled_js(num_statements=10_000)
    builder = ASTChunkBuilder(max_chunk_size=2048, language="javascript", metadata_template="default")
    chunks = builder.chunkify(content, repo_level_metadata={"filepath": "bundle.js"})

    assert len(chunks) > 3  # must be more than the 3 top-level functions
    _assert_all_chunks_within_limit(chunks, 2048)


@pytest.mark.parametrize("max_chunk_size", [500, 1000, 2048])
def test_oversized_leaf_node_various_limits(max_chunk_size):
    """Post-split works across different max_chunk_size values."""
    content = _make_bundled_js(num_statements=5_000)
    builder = ASTChunkBuilder(max_chunk_size=max_chunk_size, language="javascript", metadata_template="default")
    chunks = builder.chunkify(content, repo_level_metadata={"filepath": "bundle.js"})

    assert len(chunks) > 0
    _assert_all_chunks_within_limit(chunks, max_chunk_size)


def test_small_js_is_not_post_split():
    """Small JS that fits within limits should not be split further than AST chunking does."""
    content = _make_bundled_js(num_statements=10)
    builder = ASTChunkBuilder(max_chunk_size=2048, language="javascript", metadata_template="default")
    chunks = builder.chunkify(content, repo_level_metadata={"filepath": "small.js"})

    assert len(chunks) >= 1
    _assert_all_chunks_within_limit(chunks, 2048)


def test_post_split_metadata_line_numbers():
    """Sub-chunks from a single-line split should share the same start_line_no."""
    content = _make_bundled_js(num_statements=10_000)
    builder = ASTChunkBuilder(max_chunk_size=2048, language="javascript", metadata_template="default")
    chunks = builder.chunkify(content, repo_level_metadata={"filepath": "bundle.js"})

    # Find sub-chunks containing filler content but not function declarations
    inline_chunks = [
        c for c in chunks
        if "var v" in c["content"] and "function" not in c["content"]
    ]
    assert len(inline_chunks) > 1, "Expected multiple sub-chunks from the inline string"

    # All sub-chunks from the single-line string should share the same start_line_no
    start_lines = {c["metadata"]["start_line_no"] for c in inline_chunks}
    assert len(start_lines) == 1, (
        f"All sub-chunks from single-line split should share start_line_no, got: {sorted(start_lines)}"
    )


def test_swebench_lite_post_split_preserves_line_numbers():
    """For swebench-lite template, post-split should extract start line from _id."""
    content = _make_bundled_js(num_statements=10_000)
    builder = ASTChunkBuilder(max_chunk_size=2048, language="javascript", metadata_template="coderagbench-swebench-lite")
    chunks = builder.chunkify(
        content,
        repo_level_metadata={"instance_id": "test-instance", "filename": "bundle.js"},
    )

    assert len(chunks) > 3
    _assert_all_chunks_within_limit(chunks, 2048)

    # All chunks should have _id with the correct format
    for chunk in chunks:
        assert "_id" in chunk
        assert chunk["_id"].startswith("test-instance_")

    # Find sub-chunks from the inline string (line 6 in the generated JS, 0-indexed: row 5)
    inline_chunks = [c for c in chunks if "var v" in c["text"] and "function" not in c["text"]]
    assert len(inline_chunks) > 1

    # All inline sub-chunks should reference the same start line (extracted from original _id)
    start_lines = set()
    for c in inline_chunks:
        # _id format: "base_id_startline-endline-pieceindex"
        line_part = c["_id"].rsplit("_", 1)[1]
        start_line = int(line_part.split("-")[0])
        start_lines.add(start_line)
    assert len(start_lines) == 1, (
        f"All sub-chunks from single-line split should share start line in _id, got: {sorted(start_lines)}"
    )
    # The start line should not be 0 (which would indicate the bug)
    assert 0 not in start_lines, "start line should not default to 0"


def test_post_split_ids_are_unique():
    """All _id values must be unique after post-splitting."""
    content = _make_bundled_js(num_statements=10_000)
    builder = ASTChunkBuilder(max_chunk_size=2048, language="javascript", metadata_template="coderagbench-swebench-lite")
    chunks = builder.chunkify(
        content,
        repo_level_metadata={"instance_id": "test-instance", "filename": "bundle.js"},
    )

    ids = [c["_id"] for c in chunks]
    assert len(ids) == len(set(ids)), (
        f"Duplicate _id values found: {[x for x in ids if ids.count(x) > 1]}"
    )
