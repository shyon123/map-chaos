# Runner ↔ target contract

The host-side runner (`bench/runner.py`) and the in-sandbox entrypoint (`targets/entry.py`) talk only through files and exit codes. That keeps both frameworks behind one interface.

## Per-run layout (host temp dir, bind-mounted)
| Host path | Container path | Mode | Contents |
|---|---|---|---|
| `<run>/scratch/` | `/scratch` | rw | fixture from `build_fixture(task_id)`; the only tree MCP servers may touch |
| `<run>/in/` | `/run/in` | ro | `config.json` (RunConfig), `plan.json` (FaultPlan) |
| `<run>/out/` | `/run/out` | rw | `result.json` (EntryResult), `traces/<agent>-<server>.jsonl` |

The rootfs is read-only and `/tmp` is a tmpfs (64 MB). No MCP server is rooted outside `/scratch`. The agents have no shell or file tools besides the MCP ones, so they can't reach `/run/out`. This is documented as an assumption in SECURITY.md.

## Entrypoint
`python -m mcp_chaos.targets.entry --config /run/in/config.json --out /run/out` with env:
- `ANTHROPIC_BASE_URL=http://gw:8080/r/<run_id>` and `ANTHROPIC_API_KEY=dummy-not-a-key`
- `MCP_CHAOS_SCRATCH=/scratch`

It builds one proxy-wrapped MCP server command per (worker, server):
`[python, -m, mcp_chaos.bench.cli, proxy, --agent, W, --server, S, --plan, /run/in/plan.json, --run-id, ID, --allow, <worker allowlist>, --trace-out, /run/out/traces/W-S.jsonl, --, <upstream cmd>]`
It then runs the framework named in `config.framework` and writes `EntryResult`:
`{schema_version, run_id, status: completed|agent_error|timeout, final_answer, error: str|null, agent_turns: int, wall_s}`.
Exit code 0 means `result.json` was written, whatever the agent outcome. Non-zero means a harness error.

## Upstream commands (inside the image)
- files: `node /opt/mcp/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js /scratch`
- data: `python -m mcp_chaos.servers.sqlite_server --db /scratch/db.sqlite`
- vcs: `mcp-server-git --repository /scratch`

## Worker allowlists (least privilege)
Defined in `targets/workers.py`, shared by both frameworks:
- files: read/list/search/write/edit within `/scratch`; no move or delete outside the manifest-driven task (delete is allowed only because T6 needs it)
- data: `list_tables, describe_table, read_query, write_query`
- vcs: `git_status, git_log, git_diff*, git_show, git_add, git_commit`; no `git_reset`, `git_checkout`, or branch deletion

## Network
The target container sits only on the Docker network `mcpchaos-internal` (`--internal`). A sidecar `mcpchaos-gw` (socat) on that network and the default bridge forwards `gw:8080` to the host gateway. Nothing else is reachable.

## Local mode (no Docker)
`runner --local` runs the same entrypoint as a host subprocess with the same directory layout under a temp dir, and with upstream commands resolved on the host. It's used by integration tests (fake models, $0) and for development. It is never used for benchmark numbers.
