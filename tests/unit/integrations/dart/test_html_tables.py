"""Grid reader tests for DART document blocks (offline markup only)."""
from __future__ import annotations


def test_rowspan_header_aligns_the_sub_header_row() -> None:
    from src.integrations.dart.html_tables import read_blocks

    markup = (
        "<table>"
        '<tr><th rowspan="2">계정</th><th colspan="2">제 51 기 3분기</th><th colspan="2">제 50 기 3분기</th></tr>'
        "<tr><th>3개월</th><th>누적</th><th>3개월</th><th>누적</th></tr>"
        "</table>"
    )

    grid = read_blocks(markup)[0].grid

    assert grid[1] == ("계정", "3개월", "누적", "3개월", "누적")
    assert all(len(row) == len(grid[0]) for row in grid)


def test_rowspan_repeats_the_label_across_three_rows() -> None:
    from src.integrations.dart.html_tables import read_blocks

    markup = (
        "<table>"
        '<tr><td rowspan="3">라벨</td><td>a</td></tr>'
        "<tr><td>b</td></tr>"
        "<tr><td>c</td></tr>"
        "</table>"
    )

    grid = read_blocks(markup)[0].grid

    assert grid == (("라벨", "a"), ("라벨", "b"), ("라벨", "c"))


def test_colspan_repeats_the_header_text_in_adjacent_slots() -> None:
    from src.integrations.dart.html_tables import read_blocks

    grid = read_blocks('<table><tr><th COLSPAN="2">제 51 기 3분기</th></tr></table>')[0].grid

    assert grid == (("제 51 기 3분기", "제 51 기 3분기"),)


def test_te_and_tu_cells_are_read_into_the_grid() -> None:
    from src.integrations.dart.html_tables import read_blocks

    grid = read_blocks("<table><tr><TE>1,000</TE><TU>2,000</TU></tr></table>")[0].grid

    assert grid == (("1,000", "2,000"),)


def test_blocks_keep_document_order_of_paragraphs_and_tables() -> None:
    from src.integrations.dart.html_tables import read_blocks

    markup = "<p>first</p><table><tr><td>1</td></tr></table><p>second</p><table><tr><td>2</td></tr></table>"

    blocks = read_blocks(markup)

    assert [(block.kind, block.text, block.grid) for block in blocks] == [
        ("text", "first", ()),
        ("table", "", (("1",),)),
        ("text", "second", ()),
        ("table", "", (("2",),)),
    ]


def test_unclosed_table_is_closed_at_end_of_input() -> None:
    from src.integrations.dart.html_tables import read_blocks

    blocks = read_blocks("<table><tr><td>1</td><td>2")

    assert len(blocks) == 1
    assert blocks[0].kind == "table"
    assert blocks[0].grid == (("1", "2"),)


def test_nested_table_flattens_into_the_parent_cell_text() -> None:
    from src.integrations.dart.html_tables import read_blocks

    markup = "<table><tr><td>a<table><tr><td>inner</td></tr></table>b</td><td>2</td></tr></table>"

    assert read_blocks(markup)[0].grid == (("ainnerb", "2"),)


def test_carriage_returns_and_whitespace_collapse_to_single_spaces() -> None:
    from src.integrations.dart.html_tables import read_blocks

    blocks = read_blocks("  \n<p>재무&cr;상태표   \n 표</p>  ")

    assert [block.text for block in blocks] == ["재무 상태표 표"]


def test_malformed_markup_never_raises() -> None:
    from src.integrations.dart.html_tables import read_blocks

    assert read_blocks("") == ()
    assert [(block.kind, block.text) for block in read_blocks("</table></td></tr><td>orphan")] == [("text", "orphan")]
    assert read_blocks("<table></table>") == ()
    assert read_blocks('<table><tr><td COLSPAN="abc">v</td></tr></table>')[0].grid == (("v",),)


def test_empty_row_adds_no_columns_and_rows_pad_to_the_grid_width() -> None:
    from src.integrations.dart.html_tables import read_blocks

    grid = read_blocks("<table><tr></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>")[0].grid

    assert grid == (("", "", ""), ("1", "2", "3"))


def test_unclosed_cells_and_implicit_rows_stay_in_the_grid() -> None:
    from src.integrations.dart.html_tables import read_blocks

    assert read_blocks("<table><tr><td>a<td>b</td></tr></table>")[0].grid == (("a", "b"),)
    assert read_blocks("<table><td>solo</td></table>")[0].grid == (("solo",),)
