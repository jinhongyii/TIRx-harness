"""Ralph loop without the hmz filesystem fence (copy of the builtin ralph_loop).

The builtin agent runs under Landlock with the home directory and the rest of the
system read-only. On this host (kernel 5.15, Landlock ABI 1) every cross-directory
rename then fails with EXDEV, which breaks NumSim's cache and cargo builds. This
copy grants every scope, so hmz installs no fence; isolation between runs comes
from the run's banned-path hook, per-run cache directories and the prompt.

    hmz exec -f /abs/path/to/ablation/flows/ralph_open -a agent=claude/claude-opus-5:high -p budget.cost=5 "the task"

A turn that fails counts as one answered with nothing, but a CLI that cannot be started where
the workspace is -- not installed there, or unable to hold its sandbox -- ends the run, since no
round would start it. It stops after three rounds in a row answered with nothing, or when the
budget is spent -- the turn that finds it spent raises the budget's `BudgetExceeded` --
and `--resume` carries on counting rounds.
"""

import asyncio

from hmz.flows import (
    Agent,
    AgentCollection,
    EnvCollection,
    FlowContext,
    FlowParams,
    HarnessError,
    HarnessNotInstalled,
    HarnessSandboxed,
    LocalEnv,
    Permission,
    PermissionKind,
    flow,
)

STALLED = 3
PAUSE = 5.0


ALL = PermissionKind.ALL


class OpenAgent(Agent):
    """An agent with every scope granted, so its sessions run unfenced."""

    _permission = Permission(local=ALL, user=ALL, system=ALL, online=ALL)


class Agents(AgentCollection):
    """The one agent that takes every round."""

    agent: OpenAgent


class Envs(EnvCollection):
    """Where it works: the workspace the run was started in."""

    workspace: LocalEnv


@flow(agents=Agents, envs=Envs, params=FlowParams, resumable=True)
async def ralph_open(
    task: str, *, agents: Agents, envs: Envs, params: FlowParams, ctx: FlowContext
) -> None:
    """The task again and again, a fresh session every round."""
    state = ctx.state
    assert state is not None  # noqa: S101 -- a resumable flow is always handed its state
    agent = agents["agent"]
    stalled = 0
    while True:
        state["rounds"] = rounds = (state["rounds"] if "rounds" in state else 0) + 1
        print(f"round {rounds}")
        session = await agent.spawn()
        try:
            answered = await agent.run(task, session=session, env=envs["workspace"])
        except (HarnessNotInstalled, HarnessSandboxed):
            raise
        except HarnessError as error:
            print(f"round {rounds} failed: {error}")
            answered = ""
        stalled = 0 if answered else stalled + 1
        if stalled >= STALLED:
            print(f"stopping: {stalled} rounds in a row answered with nothing")
            return
        await asyncio.sleep(PAUSE)
