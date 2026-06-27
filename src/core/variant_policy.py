"""Canonical ablation-variant policy table.

Single source of truth shared by:
  - agent_pipeline.py — applies the policy to configure a run (arch flags, retrieval
    strategy, snippet tools) and records it in the run summary / runtime-config lock.
  - eval.py — records the same policy as run metadata.

These two used to carry duplicated copies of this table and drifted out of sync
(e.g. eval.py was missing the snippet-tools and budgeted-retrieval variants).
Keeping the table here, imported by both, makes that drift structurally impossible.

The static invariants checker (scripts/check_variant_invariants.py) AST-parses the
``VARIANT_POLICY_TABLE`` literal in this module, so it must remain a module-level
dict literal.
"""

VARIANT_POLICY_TABLE: dict[str, dict] = {
    # ── Retrieval phase (runs BEFORE the architecture ladder) ────────────────────
    # Retrieval strategies are compared on the flat_baseline architecture (no L2/L3
    # layers), so the only varied factor is retrieval. The winner (R*) is then frozen
    # into the architecture-ladder configs. Hence all ab_retrieval_* variants here
    # carry the SAME flat architecture (exploration/synthesis/validation/scratchpads
    # OFF) and differ only in retrieval_strategy.
    "ab_retrieval_iterative_react": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_react_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
    },
    "ab_retrieval_one_shot_fingerprint": {
        "phase2_anchor": False,
        "repo_context_source": "one_shot_fingerprint_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "one_shot_fingerprint",
    },
    "ab_retrieval_one_shot_fingerprint_budgeted": {
        "phase2_anchor": False,
        "repo_context_source": "one_shot_fingerprint_budgeted_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "one_shot_fingerprint_budgeted",
    },
    "ab_retrieval_neural_embedding": {
        "phase2_anchor": False,
        "repo_context_source": "neural_embedding_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "neural_embedding",
    },
    "ab_retrieval_bm25": {
        "phase2_anchor": False,
        "repo_context_source": "bm25_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "bm25",
    },
    # Repair-feature variants sit on the ARCHITECTURE CHAMPION (validation: L1+L2+L3,
    # scratchpads OFF) — NOT full_system. Per the Phase-3 result A* the scratchpad layer
    # is net-harmful (over-engineering), so measuring repair features on a full_system base
    # would confound the repair effect with scratchpad damage. validation is the champion.
    "ab_stateful_tree_off": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "iterative_exploration_synthesis_validation",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
        "stateful_tree_enabled": False,
        "prev_attempt_context_enabled": True,
    },
    "ab_stateful_tree_on": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "iterative_exploration_synthesis_validation",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
        "stateful_tree_enabled": True,
        "prev_attempt_context_enabled": True,
    },
    # AB-17 all-on baseline: history AND tree. (Previously tree was False, making this
    # identical to ab_stateful_tree_off (AB-19) — Check-3 compared two identical configs
    # and measured nothing. Fixed: tree True so AB-17 is the genuine all-on arm.)
    "ab_prev_attempt_ctx_on": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "iterative_exploration_synthesis_validation",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
        "stateful_tree_enabled": True,
        "prev_attempt_context_enabled": True,
    },
    "ab_prev_attempt_ctx_off": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "iterative_exploration_synthesis_validation",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
        "stateful_tree_enabled": False,
        "prev_attempt_context_enabled": False,
    },
    "full_system": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_react",
    },
    "ab_snippet_tools_baseline": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "staged_pipeline",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "one_shot_fingerprint",
        "snippet_tools_enabled": True,
    },
    "ab_snippet_tools_on": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_react",
        "snippet_tools_enabled": True,
    },
    "ab_snippet_tools_off": {
        "phase2_anchor": False,
        "react_repair_enabled": True,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_react",
        "snippet_tools_enabled": False,
    },
    "validation": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_synthesis_validation",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
    },
    "synthesis": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_plus_synthesis",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
    },
    "exploration": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_react",
    },
    "one_shot_direct": {
        "phase2_anchor": False,
        "repo_context_source": "static_fingerprint_only",
        "classification_required": False,
        "repair_enabled": False,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "none",
    },
}


# Policy for any variant not listed above (e.g. flat_baseline): a plain staged
# pipeline with no architecture layers.
FALLBACK_VARIANT_POLICY: dict = {
    "phase2_anchor": True,
    "repo_context_source": "staged_pipeline",
    "classification_required": True,
    "repair_enabled": True,
    "exploration_enabled": False,
    "synthesis_enabled": False,
    "validation_enabled": False,
    "scratchpads_enabled": False,
    "retrieval_strategy": "one_shot_fingerprint",
    # Baseline repair is the single-shot loop (current Dockerfile + build log only);
    # the ReAct repair agent is a gated component enabled only by specific variants.
    "react_repair_enabled": False,
}


def resolve_variant_policy(variant: str) -> dict:
    """Return the policy for ``variant`` (a fresh copy), or the fallback policy."""
    return dict(VARIANT_POLICY_TABLE.get(variant, FALLBACK_VARIANT_POLICY))


def variant_intrinsic_skips(variant: str) -> dict:
    """Phases a variant SKIPS by its own nature, independent of any CLI ``--skip-*`` flag.

    Single source of truth shared by:
      - agent_pipeline.determine_phase_skips — gates the pipeline's execution path.
      - eval.py status accounting — so a stage a variant legitimately omits is recorded
        as "skipped", never penalised as "missing" (which would mislabel an otherwise
        successful run, e.g. one_shot_direct, as PARTIAL).

    classify follows the policy's ``classification_required``; one_shot_direct additionally
    omits the post-generation validation gate and the install guide (intrinsic to the
    static one-shot variant, with no policy key of their own).
    """
    policy = resolve_variant_policy(variant)
    return {
        "classify": not policy.get("classification_required", True),
        "validation_gate": variant == "one_shot_direct",
        "install_guide": variant == "one_shot_direct",
    }
