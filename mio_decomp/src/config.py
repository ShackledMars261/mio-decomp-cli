from pathlib import Path

import typer
from pydantic import BaseModel, Field


class ConfigModel(BaseModel):
    game_dir: Path = Field(
        default=Path("C:/Program Files (x86)/Steam/steamapps/common/MIO").resolve(),
    )


class ConfigManager:
    def __init__(self, typer_app_dir: str) -> None:
        self.typer_app_dir: Path = Path(typer_app_dir).resolve()
        self.config_path: Path = self.typer_app_dir / "config.json"

        if self.config_path.exists():
            self.load_from_file()
        else:
            self.config_model: ConfigModel = ConfigModel()
            self.save_to_file()

    def save_to_file(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.touch(exist_ok=True)
        with self.config_path.open("w") as f:
            f.write(self.config_model.model_dump_json(indent=4))

    def load_from_file(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.touch(exist_ok=True)
        with self.config_path.open("r") as f:
            self.config_model = ConfigModel.model_validate_json(json_data=f.read())

    def set_value_from_key(self, key: str, value: str) -> None:
        if key not in self.config_model.model_dump().keys():
            raise AttributeError
        setattr(self.config_model, key, value)
        self.save_to_file()

    def get_value_from_key(self, key: str) -> str:
        if key not in self.config_model.model_dump().keys():
            raise AttributeError
        return str(self.config_model.model_dump()[key])

    def reset_config(self) -> None:
        self.config_model = ConfigModel()
        self.save_to_file()


config: ConfigManager = ConfigManager(typer.get_app_dir("mio-decomp"))

if __name__ == "__main__":
    manager = ConfigManager(typer_app_dir=".")
    manager.load_from_file()
    manager.save_to_file()
