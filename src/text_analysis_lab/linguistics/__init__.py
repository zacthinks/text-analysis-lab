"""Linguistic model and lexical support for TeAL."""

from text_analysis_lab.linguistics.heads import (
    SemanticHeadRules,
    export_default_semantic_head_rules,
    get_default_semantic_head_rules,
    load_semantic_head_rules,
    resolve_semantic_head_indices,
)

__all__ = [
    "SemanticHeadRules",
    "export_default_semantic_head_rules",
    "get_default_semantic_head_rules",
    "load_semantic_head_rules",
    "resolve_semantic_head_indices",
]
