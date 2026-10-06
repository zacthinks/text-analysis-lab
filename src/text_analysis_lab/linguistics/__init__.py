"""Linguistic model and lexical support for TeAL."""

from text_analysis_lab.linguistics.heads import (
    SemanticHeadMove,
    SemanticHeadResolution,
    SemanticHeadRules,
    export_default_semantic_head_rules,
    get_default_semantic_head_rules,
    load_semantic_head_rules,
    resolve_semantic_head_indices,
    resolve_semantic_heads,
)

__all__ = [
    "SemanticHeadMove",
    "SemanticHeadResolution",
    "SemanticHeadRules",
    "export_default_semantic_head_rules",
    "get_default_semantic_head_rules",
    "load_semantic_head_rules",
    "resolve_semantic_head_indices",
    "resolve_semantic_heads",
]
