from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field, conint
from rich import print  # noqa: F401

from .constants import MAX_UINT64

u32 = Annotated[int, conint(strict=True, ge=0, le=4294967295)]
u64 = Annotated[int, Field(ge=0, le=MAX_UINT64)]


class Flags(StrEnum):
    Acquired = "Acquired"
    Equipped = "Equipped"


class f32x2(BaseModel):
    x: float
    y: float

    def __init__(self, x: int | float, y: int | float, **kwargs) -> None:
        super(f32x2, self).__init__(x=x, y=y, **kwargs)


class f32x3(BaseModel):
    x: float
    y: float
    z: float

    def __init__(
        self, x: int | float, y: int | float, z: int | float, **kwargs
    ) -> None:
        super(f32x3, self).__init__(x=x, y=y, z=z, **kwargs)


class Trail(BaseModel):
    points: list = []


class Marker(BaseModel):
    pos: f32x2 = f32x2(0.000000, 0.000000)
    type: u32 = 0
    placed: bool = False


class Markers(BaseModel):
    markers: list[Marker] = [Marker() for _ in range(16)]


class MapManager(BaseModel):
    displayed: bool = False


class MIOParts(BaseModel):
    map_manager: MapManager = MapManager()


class Enum_single(StrEnum):
    UP = "Up"
    CUVE = "Cuve"
    START = "Start"
    PERSONAL_ASSISTANT = "Personal_assistant"
    UNKNOWN = "Unknown"
    INTRO = "Intro"
    INTRO_2 = "Intro_2"
    FIRST_ENCOUNTER = "First_encounter"
    RAMBLING = "Rambling"
    CITY = "City"
    HOME = "Home"
    HOUSE2 = "House2"
    HOUSE3 = "House3"
    CONNECTED_WITH_PEARLS = "Connected_with_pearls"
    NEW_HOME = "New_home"
    KNOWN = "Known"
    MEL_RETURNED = "Mel_returned"
    LIBRARY = "Library"
    AFTERMATH = "Aftermath"
    RETURNED = "Returned"
    PROJECT = "Project"
    HUB = "Hub"
    CAPTURED = "Captured"
    ARENA_HINT = "Arena_hint"
    AFTER_INVASION = "After_invasion"
    VAULTS = "Vaults"
    HUB_LIGHTS_ON = "Hub_lights_on"
    CAPUCINE = "Capucine"
    WRITER = "Writer"


class HalynAlign(BaseModel):
    configuration_discrete: int = 2
    first_rotation_ever: bool = True
    state: Enum_single = Enum_single.UP


class Tomo(BaseModel):
    quest: Enum_single = Enum_single.CUVE


class FrailPuppet(BaseModel):
    quest: Enum_single = Enum_single.START


class Capu(BaseModel):
    capu_name: Enum_single = Enum_single.PERSONAL_ASSISTANT
    quest: Enum_single = Enum_single.START


class Hacker(BaseModel):
    hacker_name: Enum_single = Enum_single.UNKNOWN
    met_at_least_once: bool = False


class Philosopher(BaseModel):
    repaired_tuner: bool = False
    tuner_interacted: bool = False


class Shii(BaseModel):
    quest: Enum_single = Enum_single.INTRO
    still_dead_when_hub_attacked: bool = False
    bridge_intro_done: bool = False


class Minions(BaseModel):
    sin: bool = False
    cos: bool = False
    tan: bool = False


class Mel(BaseModel):
    quest: Enum_single = Enum_single.INTRO
    name: Enum_single = Enum_single.UNKNOWN
    minions: Minions = Minions()
    met_in_shop_once: bool = False
    rambo_awoken: bool = False


class Cos(BaseModel):
    encountered: bool = False
    cos_keepers_spawned: bool = False
    cos_keepers_killed: bool = False


class Tan(BaseModel):
    fan_stopped: bool = False


class Rad(BaseModel):
    quest: Enum_single = Enum_single.FIRST_ENCOUNTER


class Estrogen(BaseModel):
    quest: Enum_single = Enum_single.RAMBLING
    has_refused_audience: bool = False


class MIOPlotPoints(BaseModel):
    death_after_hub: u32 = 0


class HalynPlotPoints(BaseModel):
    encountered: bool = False
    statue: Enum_single = Enum_single.CITY


class Goliath(BaseModel):
    in_observatory: bool = False
    room_entrance_triggers: list[str] = ["" for _ in range(12)]


class PlotPoints(BaseModel):
    tomo: Tomo = Tomo()
    frail_puppet: FrailPuppet = FrailPuppet()
    capu: Capu = Capu()
    hacker: Hacker = Hacker()
    philosopher: Philosopher = Philosopher()
    shii: Shii = Shii()
    mel: Mel = Mel()
    cos: Cos = Cos()
    tan: Tan = Tan()
    rad: Rad = Rad()
    estragon: Estrogen = Estrogen()
    mio: MIOPlotPoints = MIOPlotPoints()
    halyn: HalynPlotPoints = HalynPlotPoints()
    goliath: Goliath = Goliath()


class Factorio(BaseModel):
    selected_experiment: u32 = 4


class MapState(BaseModel):
    face_inside_bits: list[u32] = [0 for _ in range(1241)]
    edge_visited_bits: list[u32] = [0 for _ in range(1408)]


class Trinket(BaseModel):
    equip_order: u32 = 0


class Rebuild(BaseModel):
    step: u32 = 1
    scrap_investment: u32 = 0


class RebuildNPC(BaseModel):
    scrap_investment: u32 = 0


class DatapadFlags(StrEnum):
    pass # No known datapad flags currently exist, but this is here for future compatibility and to avoid issues with the save file format if they are added in the future.


class Datapad(BaseModel):
    status: list[DatapadFlags] = []
    discovery_index: int = 0
    mark_as_read: bool = False


class PairValue(BaseModel):
    flags: list[Flags] = [Flags.Acquired]
    count: int = 0
    trinket: Trinket | None = None
    rebuild: Rebuild | None = None
    rebuild_npc: RebuildNPC | None = None
    datapad: Datapad | None = None


class Pair(BaseModel):
    key: str = ""
    value: PairValue = PairValue()


class SavedEntries(BaseModel):
    pairs: list[Pair] = [Pair() for _ in range(1710)]


class Save(BaseModel):
    flags: list[Flags] = []
    version: u32 = 5
    id: u64 = 0
    checkpoint_id: str = Field(
        default="cp_FOCUS_dispatch_zone",
    )
    checkpoint_world_pos: f32x3 = f32x3(-1000.000000, 4426.000000, 0.000000)
    checkpoint_wrap_index: int = 0
    checkpoint_is_temporary: bool = False
    previous_checkpoint_id: str = ""
    previous_checkpoint_world_pos: f32x3 = f32x3(0.000000, 0.000000, 0.000000)
    previous_checkpoint_wrap_index: int = 0
    trail: Trail = Trail()
    markers: Markers = Markers()
    mio_parts: MIOParts = MIOParts()
    halyn_align: HalynAlign = HalynAlign()
    map_trace_flags: list[Flags] = []
    plotpoints: PlotPoints = PlotPoints()
    nextfest_demo_time_to_bad_ending: float = -1.0
    nextfest_demo_time_to_good_ending: float = -1.0
    orb_slash_slot: str = ""
    factorio: Factorio = Factorio()
    nacre_in_hub_basin: u32 = 0
    nacre_buffered_in_hub_basin: u32 = 0
    shield_decay_mask: u32 = 1
    mio_wrap_index: int = 0
    map_state: MapState = MapState()


class SavedVisibility2(BaseModel):
    pairs: list = []


class SavedNotImportant(BaseModel):
    playtime: float = 0.0
    last_save_time: float = 0.0
    liquid_nacres_count: u32 = 0
    solidify_nacre_count: int = 0


class MIOSave(BaseModel):
    save: Save = Save()
    saved_entries: SavedEntries = SavedEntries()
    saved_visibility2: SavedVisibility2 = SavedVisibility2()
    saved_not_important: SavedNotImportant = SavedNotImportant()


class SaveParser:
    def __init__(self):
        self.save: MIOSave = MIOSave()

    def __convert_value(self, value: str, to_json: bool = True, key: str = None) -> int | float | str | bool | f32x2 | f32x3 | Enum_single:
        """
        Converts a value from the save file format to a Python type, or vice versa.
        
        Args:
            value (String): The value to convert.
            to_json (bool): Whether to convert to JSON or to save file. Default True.
            key (str): Used in determining u32 vs i32. Only when converting to save file.

        Returns:
            value in corresponding type, or
            str: save file format if to_json is False.
        """
        if to_json:
            type: str
            content: str

            type, content = value[:-1].split("(", 1)
            content = content.strip('"')
            match type:
                case "i32":
                    return int(content)
                case "u32":
                    return int(content)
                case "u64":
                    return int(content)
                case "String":
                    return content
                case "bool":
                    return content == "true"
                case "f32x3":
                    return f32x3(*[float(n) for n in content.split(", ")])
                case "f32x2":
                    return f32x2(*[float(n) for n in content.split(", ")])
                case "f32":
                    return float(content)
                case "f64":
                    return float(content)
                case "Enum_single":
                    return Enum_single(content)
                case "Flags":
                    return [Flags(flag) for flag in content.split('""') if flag]
                case _:
                    typer.Abort()
                    return None
        else:
            if isinstance(value, bool):
                return f"bool({str(value).lower()})"
            if isinstance(value, int):
                if key is None:
                    return f"i32({value})"
                if key == "id":
                    return f"u64({value})"
                if key in ["version", "plotpoints.mio.death_after_hub", "factorio.selected_experiment", "nacre_in_hub_basin",  "nacre_buffered_in_hub_basin", "shield_decay_mask", "liquid_nacres_count"]:
                    return f"u32({value})"
                keysplit: list[str] = key.split(".")
                if len(keysplit) > 1 and keysplit[1] in ["markers", "face_inside_bits", "edge_visited_bits"]:
                    return f"u32({value})"
                if len(keysplit) > 3 and keysplit[3] in ["rebuild", "rebuild_npc", "trinket"]:
                    return f"u32({value})"
                return f"i32({value})"
            if isinstance(value, float):
                if key == "last_save_time":
                    return f"f64({value:.6f})"
                return f"f32({value:.6f})"
            if isinstance(value, Enum_single):
                return f"Enum_single(\"{value.value}\")"
            if isinstance(value, f32x2):
                vals = ", ".join(f"{v:.6f}" for v in value.model_dump().values())
                return f"f32x2({vals})"
            if isinstance(value, f32x3):
                vals = ", ".join(f"{v:.6f}" for v in value.model_dump().values())
                return f"f32x3({vals})"
            if isinstance(value, str):
                return f"String(\"{value}\")"
            return str(value)

    def __safe_set_value_by_key(self, group: str, key: str, value: str) -> None:
        if value.startswith("Array"):
            return

        access_string: str = f"{group.lower()}.{key}"
        current_object = self.save
        parts = access_string.split(".")
        for part in parts[:-1]:
            if part.isdigit():
                current_object = current_object[int(part)]  # ty:ignore[not-subscriptable]
            else:
                if getattr(current_object, part) is None:
                    match part:
                        case "datapad":
                            setattr(current_object, "datapad", Datapad())
                        case "rebuild":
                            setattr(current_object, "rebuild", Rebuild())
                        case "rebuild_npc":
                            setattr(current_object, "rebuild_npc", RebuildNPC())
                        case "trinket":
                            setattr(current_object, "trinket", Trinket())
                        case _:
                            print(f"NONE FOUND! {part}")

                current_object = getattr(current_object, part)

        processed_value = self.__convert_value(value)
        if value is None:
            typer.Abort()

        if isinstance(current_object, list):
            current_object[int(parts[-1])] = processed_value
        else:
            setattr(current_object, parts[-1], processed_value)

    def parse_save(self, input_path: Path) -> str:
        """Parses a MIO save file into JSON.

        Args:
            input_path (Path): Path to the save_file.

        Returns:
            str: The JSON representing the save file.
        """
        lines: list[str] = [
            line.strip()
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if not line == ""
        ]

        grouped_lines: dict[str, list[str]] = {}

        current_group: str | None = None

        for line in lines:
            line = line.strip()
            if current_group is None:
                if line.endswith("{"):
                    current_group = line[:-2]
                    grouped_lines[current_group] = []
                    continue
            else:
                if line == "}":
                    current_group = None
                    continue
                grouped_lines[current_group].append(line)

        for group, lines in grouped_lines.items():
            for line in lines:
                key, _, value = line.split(" ", 2)
                self.__safe_set_value_by_key(group, key, value)

        return self.save.model_dump_json(indent=4, exclude_none=True)
    
    def __serialize_recursive(self, obj, prefix="") -> list[str]:
        lines = []
        
        if isinstance(obj, BaseModel) and not isinstance(obj, (f32x2, f32x3)):
            for field_name, field_value in obj:
                if field_value is None:
                    continue
                new_prefix = f"{prefix}.{field_name}" if prefix else field_name
                lines.extend(self.__serialize_recursive(field_value, new_prefix))
        
        elif isinstance(obj, list):
            if prefix == "pairs" and len(obj) > 1:
                active_items = [obj[0]] + [item for item in obj[1:] if item not in [None, ""] and item.key != ""]
            else:
                active_items = [item for item in obj if item not in [None, ""]]

            if prefix.endswith("flags") or prefix.endswith("datapad.status"):
                # print([flag.value for flag in obj])
                lines.append(f'{prefix} = Flags({"" if len(active_items) == 0 else "".join(f"\"{flag.value.capitalize()}\"" for flag in active_items)})')
            else:
                lines.append(f'{prefix} = Array({len(active_items)})')
                for i, item in enumerate(active_items):
                    if item is None:
                        continue
                    lines.extend(self.__serialize_recursive(item, f"{prefix}.{i}"))

        else:
            formatted = self.__convert_value(obj, to_json=False, key=prefix)
            lines.append(f'{prefix} = {formatted}')
            
        return lines
    
    def compile_save(self, json_path: Path) -> str:
        """Compiles a MIO json save file back into a MIO save file.

        Args:
            input_path (Path): Path to the json save file.

        Returns:
            str: The compiled save file content.
        """
        data = json_path.read_text(encoding="utf-8")
        self.save = MIOSave.model_validate_json(data)
        
        final_output = []
        
        for block_name in ["save", "saved_entries", "saved_visibility2", "saved_not_important"]:
            block_data = getattr(self.save, block_name)
            
            final_output.append(f"{block_name.capitalize()} {{")
            
            content_lines = self.__serialize_recursive(block_data)
            final_output.extend([f"  {line}" for line in content_lines])
            
            final_output.append("}\n")

        return "\n".join(final_output) + "\n" # for some reason they always have an couple of newlines at the end

if __name__ == "__main__":
    parser: SaveParser = SaveParser()
    # parser.__safe_set_value_by_key("", "", "")
    print(parser.compile_save(Path(r"tests\test_saves\100_percent.json")))
