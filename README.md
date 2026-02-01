# mio-decomp-cli
A CLI for decompiling the .gin files from MIO: Memories in Orbit.

Huge thanks to @mistwreathed for creating the original version of this tool.

## Installation
```sh
python -m pip install mio_decomp
mio-decomp version # Verify it installed successfully
```

## Setup
```sh
mio-decomp config set game_dir "<Path to your MIO install>" # Defaults to "C:\Program Files (x86)\Steam\steamapps\common\MIO". If your install is there, you can skip this command.
```

## Use
```sh
mio-decomp decompile gin1.gin ./dir_with_many_gin_files 2gin.gin ./another_dir -o output
```