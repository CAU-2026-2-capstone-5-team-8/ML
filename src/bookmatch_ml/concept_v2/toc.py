"""Restore canonical TOC hierarchy and deterministic depth-first paths."""

from collections import defaultdict
from dataclasses import dataclass, field

from bookmatch_ml.schemas import TocEntry


class TocTreeError(ValueError):
    """TOC parent, level, or sibling order is inconsistent."""


@dataclass
class TocNode:
    entry: TocEntry
    children: list["TocNode"] = field(default_factory=list)


@dataclass(frozen=True)
class TocVisit:
    entry: TocEntry
    parent_path: tuple[str, ...]
    path: tuple[str, ...]
    traversal_position: int


@dataclass
class TocTree:
    roots: list[TocNode]
    traversal: list[TocVisit]


def reconstruct_toc(entries: list[TocEntry]) -> TocTree:
    """Reject duplicate sibling positions; parent links, not input order, define the tree."""

    by_id = {entry.toc_entry_id: TocNode(entry) for entry in entries}
    if len(by_id) != len(entries):
        raise TocTreeError("duplicate toc_entry_id")
    siblings: dict[str | None, list[TocNode]] = defaultdict(list)
    positions: set[tuple[str | None, int]] = set()
    for entry in entries:
        if entry.parent_entry_id is None:
            if entry.level != 1:
                raise TocTreeError(f"root {entry.toc_entry_id} must be level 1")
        else:
            parent = by_id.get(entry.parent_entry_id)
            if parent is None or parent.entry.book_id != entry.book_id:
                raise TocTreeError(f"orphan or cross-book parent for {entry.toc_entry_id}")
            if entry.level != parent.entry.level + 1:
                raise TocTreeError(f"inconsistent level for {entry.toc_entry_id}")
        position = (entry.parent_entry_id, entry.order_index)
        if position in positions:
            raise TocTreeError(f"duplicate sibling order_index for {entry.toc_entry_id}")
        positions.add(position)
        siblings[entry.parent_entry_id].append(by_id[entry.toc_entry_id])
    for group in siblings.values():
        group.sort(key=lambda node: (node.entry.order_index, node.entry.toc_entry_id))
    roots = siblings[None]
    traversal: list[TocVisit] = []
    seen: set[str] = set()

    def visit(node: TocNode, path: tuple[str, ...]) -> None:
        entry = node.entry
        if entry.toc_entry_id in seen:
            raise TocTreeError("TOC cycle")
        seen.add(entry.toc_entry_id)
        full_path = (*path, entry.title)
        traversal.append(TocVisit(entry, path, full_path, len(traversal)))
        node.children = siblings[entry.toc_entry_id]
        for child in node.children:
            visit(child, full_path)

    for root in roots:
        visit(root, ())
    if len(seen) != len(entries):
        raise TocTreeError("TOC contains unreachable nodes")
    return TocTree(roots, traversal)
