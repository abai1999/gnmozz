"""Task/spec registry for Coarse2Contact v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_CONFIG_ROOT = ROOT / "configs" / "coarse2contact" / "tasks"


def _as_tuple(value, *, default: tuple[Any, ...] = ()) -> tuple[Any, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class EntitySpec:
    name: str
    role: str
    primitive: str
    color_hint: Optional[str] = None
    rgb_hint: Optional[tuple[int, int, int]] = None
    observable_hints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "EntitySpec":
        rgb_hint = data.get("rgb_hint")
        if rgb_hint is not None:
            rgb_hint = tuple(int(x) for x in _as_tuple(rgb_hint))
            if len(rgb_hint) != 3:
                raise ValueError(f"EntitySpec {name!r}: rgb_hint must have 3 entries")
        return cls(
            name=name,
            role=str(data.get("role", "")),
            primitive=str(data.get("primitive", "")),
            color_hint=data.get("color_hint"),
            rgb_hint=rgb_hint,
            observable_hints=dict(data.get("observable_hints", {}) or {}),
        )


@dataclass(frozen=True)
class PrecisionSkillSpec:
    name: str
    skill_type: str
    target_entity: str
    reference_entity: str
    controlled_dofs: tuple[str, ...]
    confidence_threshold: float = 0.35
    apply_confidence: float = 0.40
    shadow_confidence: float = 0.20
    xy_tolerance: float = 0.010
    z_tolerance: float = 0.012
    yaw_tolerance: float = 0.20
    max_xy_step: float = 0.0010
    max_dz_step: float = 0.0010
    max_yaw_step: float = 0.035
    roi_size_px: int = 96
    roi_resize_px: int = 128
    heatmap_xy_range_m: float = 0.040
    heatmap_size: int = 16
    heatmap_sigma_px: float = 1.5
    heatmap_channels: int = 3
    heatmap_pos_weight: float = 8.0
    abstain_if_low_observability: bool = True
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "PrecisionSkillSpec":
        return cls(
            name=name,
            skill_type=str(data.get("skill_type", "")),
            target_entity=str(data.get("target_entity", "")),
            reference_entity=str(data.get("reference_entity", "")),
            controlled_dofs=tuple(str(x) for x in _as_tuple(data.get("controlled_dofs"), default=())),
            confidence_threshold=_as_float(data.get("confidence_threshold"), 0.35),
            apply_confidence=_as_float(data.get("apply_confidence"), _as_float(data.get("confidence_threshold"), 0.40)),
            shadow_confidence=_as_float(data.get("shadow_confidence"), 0.20),
            xy_tolerance=_as_float(data.get("xy_tolerance"), 0.010),
            z_tolerance=_as_float(data.get("z_tolerance"), 0.012),
            yaw_tolerance=_as_float(data.get("yaw_tolerance"), 0.20),
            max_xy_step=_as_float(data.get("max_xy_step"), 0.0010),
            max_dz_step=_as_float(data.get("max_dz_step"), 0.0010),
            max_yaw_step=_as_float(data.get("max_yaw_step"), 0.035),
            roi_size_px=int(data.get("roi_size_px", 96) or 96),
            roi_resize_px=int(data.get("roi_resize_px", 128) or 128),
            heatmap_xy_range_m=_as_float(data.get("heatmap_xy_range_m"), 0.040),
            heatmap_size=int(data.get("heatmap_size", 16) or 16),
            heatmap_sigma_px=_as_float(data.get("heatmap_sigma_px"), 1.5),
            heatmap_channels=int(data.get("heatmap_channels", 4) or 4),
            heatmap_pos_weight=_as_float(data.get("heatmap_pos_weight"), 8.0),
            abstain_if_low_observability=bool(data.get("abstain_if_low_observability", True)),
            notes=tuple(str(x) for x in _as_tuple(data.get("notes"), default=())),
        )


@dataclass(frozen=True)
class StageTransition:
    name: str
    owner: str
    skill_name: Optional[str]
    next_stage: Optional[str]
    transition_on: str
    min_steps: int = 0
    max_steps: int = 0
    recover_to_stage: Optional[str] = None
    gripper_mode: str = "hold"
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageTransition":
        skill_name = data.get("skill_name")
        next_stage = data.get("next_stage")
        recover_to_stage = data.get("recover_to_stage")
        return cls(
            name=str(data.get("name", "")),
            owner=str(data.get("owner", "planner")),
            skill_name=None if skill_name in (None, "", "null") else str(skill_name),
            next_stage=None if next_stage in (None, "", "null") else str(next_stage),
            transition_on=str(data.get("transition_on", "always")),
            min_steps=int(data.get("min_steps", 0) or 0),
            max_steps=int(data.get("max_steps", 0) or 0),
            recover_to_stage=None if recover_to_stage in (None, "", "null") else str(recover_to_stage),
            gripper_mode=str(data.get("gripper_mode", "hold")),
            notes=tuple(str(x) for x in _as_tuple(data.get("notes"), default=())),
        )


@dataclass(frozen=True)
class PrecisionTaskSpec:
    task_name: str
    description: str
    language_targets: tuple[str, ...]
    entities: dict[str, EntitySpec]
    skills: dict[str, PrecisionSkillSpec]
    stage_graph: tuple[StageTransition, ...]
    default_stage: str
    runtime_flags: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def stages(self) -> tuple[StageTransition, ...]:
        return self.stage_graph

    def validate(self) -> None:
        if not self.task_name:
            raise ValueError("PrecisionTaskSpec.task_name must be non-empty")
        if self.default_stage and self.default_stage not in {s.name for s in self.stage_graph}:
            raise ValueError(f"default_stage {self.default_stage!r} not found in stage graph")
        if not self.stage_graph:
            raise ValueError(f"task {self.task_name!r} must define a non-empty stage graph")
        if not self.skills:
            raise ValueError(f"task {self.task_name!r} must define at least one skill")

        for skill_name, skill in self.skills.items():
            if skill.target_entity and skill.target_entity not in self.entities:
                raise ValueError(f"skill {skill_name!r} references unknown target entity {skill.target_entity!r}")
            if skill.reference_entity and skill.reference_entity not in self.entities:
                raise ValueError(f"skill {skill_name!r} references unknown reference entity {skill.reference_entity!r}")
            if not skill.skill_type:
                raise ValueError(f"skill {skill_name!r} must define skill_type")

        stage_names = {s.name for s in self.stage_graph}
        for stage in self.stage_graph:
            if stage.skill_name and stage.skill_name not in self.skills:
                raise ValueError(f"stage {stage.name!r} references unknown skill {stage.skill_name!r}")
            if stage.next_stage and stage.next_stage not in stage_names:
                raise ValueError(f"stage {stage.name!r} references unknown next_stage {stage.next_stage!r}")
            if stage.recover_to_stage and stage.recover_to_stage not in stage_names:
                raise ValueError(f"stage {stage.name!r} references unknown recover_to_stage {stage.recover_to_stage!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrecisionTaskSpec":
        entities = {
            str(name): EntitySpec.from_dict(str(name), spec)
            for name, spec in dict(data.get("entities", {}) or {}).items()
        }
        skills = {
            str(name): PrecisionSkillSpec.from_dict(str(name), spec)
            for name, spec in dict(data.get("skills", {}) or {}).items()
        }
        stage_graph = tuple(StageTransition.from_dict(item) for item in _as_tuple(data.get("stages"), default=()))
        spec = cls(
            task_name=str(data.get("task_name", "")),
            description=str(data.get("description", "")),
            language_targets=tuple(str(x) for x in _as_tuple(data.get("language_targets"), default=())),
            entities=entities,
            skills=skills,
            stage_graph=stage_graph,
            default_stage=str(data.get("default_stage", stage_graph[0].name if stage_graph else "")),
            runtime_flags=dict(data.get("runtime_flags", {}) or {}),
            notes=tuple(str(x) for x in _as_tuple(data.get("notes"), default=())),
        )
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, path: Path) -> "PrecisionTaskSpec":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def get_stage(self, stage_name: str) -> StageTransition:
        for stage in self.stage_graph:
            if stage.name == stage_name:
                return stage
        raise KeyError(stage_name)

    def get_skill(self, skill_name: str) -> PrecisionSkillSpec:
        if skill_name not in self.skills:
            raise KeyError(skill_name)
        return self.skills[skill_name]

    def get_entity(self, entity_name: str) -> EntitySpec:
        if entity_name not in self.entities:
            raise KeyError(entity_name)
        return self.entities[entity_name]

    def skill_by_type(self, skill_type: str) -> Optional[PrecisionSkillSpec]:
        for skill in self.skills.values():
            if skill.skill_type == skill_type:
                return skill
        return None

    def skills_of_type(self, skill_type: str) -> list[PrecisionSkillSpec]:
        return [skill for skill in self.skills.values() if skill.skill_type == skill_type]

    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stage_graph)

    def stage_by_name(self, stage_name: str) -> StageTransition:
        return self.get_stage(stage_name)


class PrecisionTaskRegistry:
    def __init__(self, config_root: Path | None = None) -> None:
        self.config_root = Path(config_root or DEFAULT_TASK_CONFIG_ROOT)

    def available_tasks(self) -> list[str]:
        if not self.config_root.exists():
            return []
        return sorted(p.stem for p in self.config_root.glob("*.yaml"))

    def path_for(self, task_name: str) -> Path:
        return self.config_root / f"{task_name}.yaml"

    def load(self, task_name: str) -> Optional[PrecisionTaskSpec]:
        path = self.path_for(task_name)
        if not path.exists():
            return None
        return PrecisionTaskSpec.from_yaml(path)

    def load_or_none(self, task_name: str) -> Optional[PrecisionTaskSpec]:
        return self.load(task_name)


def load_precision_task_spec(task_name: str, *, config_root: Path | None = None) -> Optional[PrecisionTaskSpec]:
    return PrecisionTaskRegistry(config_root=config_root).load(task_name)
