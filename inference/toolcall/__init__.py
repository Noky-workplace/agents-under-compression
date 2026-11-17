"""Tool-calling evaluation against llama-server (OpenAI-compatible API).

Start llama-server separately on the GPU host, bound to 127.0.0.1, then run::

    python -m inference.toolcall.run_eval --base-url http://127.0.0.1:8080 \
        --label Q4_K_M --seeds 3 --out research/results/phase2/toolcall.jsonl
    python -m inference.toolcall.report research/results/phase2/toolcall.jsonl

Seeds are integers 0..N-1. Reusing the output resumes completed label/task/seed
combinations. Multi-turn expected calls describe the complete ordered trajectory;
scripted tool_results are a list indexed by global call order. Parallel calls may arrive in any order. Multi-turn cases require one call per
turn so dependent arguments are chosen after feedback. The JSONL is hand-maintained. Clarify tasks require nonempty text
and no calls; semantic quality of the clarification requires human review.
"""
