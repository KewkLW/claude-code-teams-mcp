"""Task Distribution Engine for claude-teams.

Extends the basic FIFO task claiming with:
- Capability matching (route tasks to the right agent type)
- Load balancing (least-loaded capable agent)
- Priority scheduling (critical path first)
- Result routing (auto-unblock, notify, trigger follow-ups)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from claude_teams._filelock import file_lock
from claude_teams.models import TaskFile, TeammateMember
from claude_teams.tasks import _tasks_dir, list_tasks, get_task, update_task, create_task
from claude_teams.teams import read_config
from claude_teams.messaging import send_plain_message, send_task_assignment

logger = logging.getLogger(__name__)


# ── Agent Capability Profiles ──────────────────────────────────────────

CAPABILITY_PROFILES: dict[str, dict[str, Any]] = {
    # Claude models
    "opus": {
        "strengths": ["architecture", "complex_reasoning", "multi_file_refactor", "code_review", "planning"],
        "speed": "slow",
        "cost": "high",
        "max_concurrent": 2,
    },
    "sonnet": {
        "strengths": ["implementation", "code_review", "testing", "documentation", "general"],
        "speed": "medium",
        "cost": "medium",
        "max_concurrent": 4,
    },
    "haiku": {
        "strengths": ["research", "quick_lookup", "code_review", "documentation", "simple_fixes"],
        "speed": "fast",
        "cost": "low",
        "max_concurrent": 8,
    },
    # Codex models
    "codex": {
        "strengths": ["implementation", "fast_iteration", "file_edits", "bash_commands", "testing"],
        "speed": "fast",
        "cost": "medium",
        "max_concurrent": 4,
    },
    # Gemini models
    "gemini": {
        "strengths": ["research", "web_search", "documentation", "analysis"],
        "speed": "medium",
        "cost": "low",
        "max_concurrent": 4,
    },
    # Default fallback
    "general": {
        "strengths": ["general"],
        "speed": "medium",
        "cost": "medium",
        "max_concurrent": 4,
    },
}

# Task type → required capabilities mapping
TASK_CAPABILITY_MAP: dict[str, list[str]] = {
    "architecture": ["architecture", "complex_reasoning", "planning"],
    "implementation": ["implementation", "file_edits", "fast_iteration"],
    "refactor": ["multi_file_refactor", "implementation"],
    "testing": ["testing", "implementation"],
    "code_review": ["code_review"],
    "research": ["research", "web_search", "quick_lookup"],
    "documentation": ["documentation"],
    "bug_fix": ["implementation", "simple_fixes", "fast_iteration"],
    "planning": ["planning", "architecture", "complex_reasoning"],
    "general": ["general"],
}


# ── Task Priority ──────────────────────────────────────────────────────

def compute_priority(task: TaskFile, all_tasks: list[TaskFile]) -> float:
    """Compute task priority score. Higher = more important.

    Factors:
    - Critical path weight: how many tasks does this block?
    - Depth weight: tasks blocked by fewer things are more ready
    - Metadata priority: explicit priority from task creator
    - Age: older pending tasks get slight boost
    """
    score = 0.0

    # Critical path: count how many tasks this blocks (recursively)
    blocked_count = len(task.blocks)
    # Add transitive blocks
    visited = set()
    queue = list(task.blocks)
    while queue:
        tid = queue.pop(0)
        if tid in visited:
            continue
        visited.add(tid)
        for t in all_tasks:
            if t.id == tid:
                queue.extend(t.blocks)
                blocked_count += len(t.blocks)
    score += blocked_count * 10.0

    # Readiness: fewer blockers = more ready = higher priority
    active_blockers = sum(
        1 for bid in task.blocked_by
        if any(t.id == bid and t.status != "completed" for t in all_tasks)
    )
    score -= active_blockers * 5.0

    # Explicit priority from metadata
    if task.metadata and "priority" in task.metadata:
        priority_map = {"critical": 100, "high": 50, "medium": 25, "low": 10}
        score += priority_map.get(task.metadata["priority"], 25)

    # Task type bonus (architecture/planning should happen first)
    if task.metadata and "task_type" in task.metadata:
        type_bonus = {
            "planning": 30, "architecture": 25, "implementation": 15,
            "testing": 10, "code_review": 5, "documentation": 0,
        }
        score += type_bonus.get(task.metadata["task_type"], 0)

    return score


# ── Capability Matching ────────────────────────────────────────────────

def get_agent_profile(member: TeammateMember) -> dict[str, Any]:
    """Get capability profile for a team member based on their model."""
    model = member.model.lower()
    for key, profile in CAPABILITY_PROFILES.items():
        if key in model:
            return profile
    return CAPABILITY_PROFILES["general"]


def agent_can_handle(member: TeammateMember, task: TaskFile) -> bool:
    """Check if an agent has the capabilities for a task type."""
    profile = get_agent_profile(member)
    task_type = (task.metadata or {}).get("task_type", "general")
    required = TASK_CAPABILITY_MAP.get(task_type, ["general"])
    return any(cap in profile["strengths"] for cap in required)


def rank_agents_for_task(
    task: TaskFile, members: list[TeammateMember], active_tasks: dict[str, int]
) -> list[tuple[TeammateMember, float]]:
    """Rank agents by suitability for a task.

    Returns list of (member, score) sorted by score descending.
    Score factors: capability match, current load, speed.
    """
    rankings: list[tuple[TeammateMember, float]] = []

    task_type = (task.metadata or {}).get("task_type", "general")
    required = TASK_CAPABILITY_MAP.get(task_type, ["general"])

    for member in members:
        if not member.is_active:
            continue

        profile = get_agent_profile(member)

        # Capability match score (0-100)
        matching_caps = sum(1 for cap in required if cap in profile["strengths"])
        cap_score = (matching_caps / max(len(required), 1)) * 100

        if cap_score == 0:
            continue  # Can't handle this task type at all

        # Load score (0-100, higher = less loaded)
        current_load = active_tasks.get(member.name, 0)
        max_concurrent = profile.get("max_concurrent", 4)
        if current_load >= max_concurrent:
            continue  # At capacity
        load_score = ((max_concurrent - current_load) / max_concurrent) * 100

        # Speed bonus
        speed_bonus = {"fast": 20, "medium": 10, "slow": 0}.get(profile["speed"], 10)

        # Cost efficiency (prefer cheaper for simple tasks)
        cost_penalty = {"high": -15, "medium": 0, "low": 10}.get(profile["cost"], 0)
        if task_type in ("research", "documentation", "simple_fixes"):
            cost_penalty *= 2  # Double cost sensitivity for simple tasks

        total = cap_score * 0.5 + load_score * 0.3 + speed_bonus + cost_penalty
        rankings.append((member, total))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


# ── Task Distribution ──────────────────────────────────────────────────

def get_active_task_counts(team_name: str, base_dir: Path | None = None) -> dict[str, int]:
    """Count in-progress tasks per agent."""
    all_tasks = list_tasks(team_name, base_dir)
    counts: dict[str, int] = {}
    for task in all_tasks:
        if task.status == "in_progress" and task.owner:
            counts[task.owner] = counts.get(task.owner, 0) + 1
    return counts


def find_ready_tasks(team_name: str, base_dir: Path | None = None) -> list[TaskFile]:
    """Find tasks that are pending and have no active blockers."""
    all_tasks = list_tasks(team_name, base_dir)
    ready = []
    for task in all_tasks:
        if task.status != "pending":
            continue
        if task.owner is not None:
            continue  # Already claimed
        # Check all blockers are completed
        blocked = False
        for bid in task.blocked_by:
            for t in all_tasks:
                if t.id == bid and t.status != "completed":
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            ready.append(task)
    return ready


def distribute_tasks(
    team_name: str,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Main distribution function. Assigns ready tasks to best-fit agents.

    Returns list of assignments made: [{task_id, agent_name, score}]
    """
    config = read_config(team_name)
    teammates = [m for m in config.members if isinstance(m, TeammateMember) and m.is_active]

    if not teammates:
        return []

    ready_tasks = find_ready_tasks(team_name, base_dir)
    if not ready_tasks:
        return []

    all_tasks = list_tasks(team_name, base_dir)
    active_counts = get_active_task_counts(team_name, base_dir)

    # Sort ready tasks by priority
    ready_tasks.sort(key=lambda t: compute_priority(t, all_tasks), reverse=True)

    assignments = []

    for task in ready_tasks:
        rankings = rank_agents_for_task(task, teammates, active_counts)

        if not rankings:
            logger.info(f"No capable agent for task {task.id} ({task.subject})")
            continue

        best_agent, score = rankings[0]

        # Assign the task
        try:
            update_task(
                team_name,
                task.id,
                status="in_progress",
                owner=best_agent.name,
                base_dir=base_dir,
            )
            # Send assignment notification
            send_task_assignment(team_name, task, assigned_by="distributor")

            # Update active counts for next iteration
            active_counts[best_agent.name] = active_counts.get(best_agent.name, 0) + 1

            assignments.append({
                "task_id": task.id,
                "subject": task.subject,
                "agent_name": best_agent.name,
                "agent_model": best_agent.model,
                "score": round(score, 1),
                "task_type": (task.metadata or {}).get("task_type", "general"),
            })

            logger.info(
                f"Assigned task {task.id} ({task.subject}) → {best_agent.name} "
                f"(score: {score:.1f})"
            )
        except Exception as e:
            logger.error(f"Failed to assign task {task.id}: {e}")

    return assignments


def suggest_decomposition(
    description: str,
) -> list[dict[str, str]]:
    """Suggest how to decompose a high-level task into subtasks.

    Returns list of suggested subtasks with task_type annotations.
    This is a heuristic — the team lead should refine.
    """
    # Keyword-based heuristic decomposition
    suggestions = []
    desc_lower = description.lower()

    # Always start with planning for complex tasks
    if any(word in desc_lower for word in ["build", "create", "implement", "develop", "design"]):
        suggestions.append({
            "subject": f"Plan: {description[:60]}",
            "description": f"Create an implementation plan for: {description}. Identify files to create/modify, dependencies, and testing strategy.",
            "task_type": "planning",
            "priority": "high",
        })

    # Implementation tasks
    if any(word in desc_lower for word in ["build", "create", "implement", "add", "write"]):
        suggestions.append({
            "subject": f"Implement: {description[:60]}",
            "description": f"Implement the core functionality: {description}",
            "task_type": "implementation",
            "priority": "medium",
        })

    # Testing
    if any(word in desc_lower for word in ["build", "create", "implement", "fix", "refactor"]):
        suggestions.append({
            "subject": f"Test: {description[:60]}",
            "description": f"Write tests for: {description}. Cover edge cases and integration points.",
            "task_type": "testing",
            "priority": "medium",
        })

    # Documentation
    if any(word in desc_lower for word in ["build", "create", "api", "public", "interface"]):
        suggestions.append({
            "subject": f"Document: {description[:60]}",
            "description": f"Write documentation for: {description}",
            "task_type": "documentation",
            "priority": "low",
        })

    # Research
    if any(word in desc_lower for word in ["research", "investigate", "explore", "compare", "evaluate"]):
        suggestions.append({
            "subject": f"Research: {description[:60]}",
            "description": description,
            "task_type": "research",
            "priority": "high",
        })

    # Code review
    if any(word in desc_lower for word in ["review", "audit", "check", "verify"]):
        suggestions.append({
            "subject": f"Review: {description[:60]}",
            "description": description,
            "task_type": "code_review",
            "priority": "medium",
        })

    # Fallback
    if not suggestions:
        suggestions.append({
            "subject": description[:80],
            "description": description,
            "task_type": "general",
            "priority": "medium",
        })

    return suggestions
