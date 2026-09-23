import itertools
import subprocess
from pathlib import Path
import zipfile
import shutil

# from music21.converter.subConverters import

from tqdm import tqdm

MUSESCORE_PATH = Path(
    "~/Applications/MuseScore-Studio-4.5.2.251141401-x86_64.AppImage"
).expanduser()

root_path = Path("~/Téléchargements/georges_collier").expanduser()
zip_path = root_path / "zip"
unzipped_path = root_path / "unzipped"
musicxml_path = root_path / "musicxml"


# Read all zip files
zip_files = sorted(zip_path.glob("*.zip"))

# Process them all
for zip_file in tqdm(zip_files):
    # Unzip them
    unzipped_dir = unzipped_path / zip_file.stem
    if not unzipped_dir.exists():
        unzipped_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(unzipped_dir)

    # Search for musicxml files
    musicxml_files = list(unzipped_dir.rglob("*.musicxml"))
    xml_files = list(unzipped_dir.rglob("*.xml"))
    mxl_files = list(unzipped_dir.rglob("*.mxl"))
    all_files = musicxml_files + xml_files + mxl_files
    all_files = list(filter(lambda f: "__MACOSX" not in f.parts, all_files))
    if len(all_files) == 0:
        if mscz_file := next(unzipped_dir.rglob("*.mscz"), None):
            new_file = (musicxml_path / zip_file.stem).with_suffix(".musicxml")
            print(f"No musicxml found for file {zip_file}, trying with the mscz file")
            process = subprocess.run(
                [
                    MUSESCORE_PATH,
                    mscz_file.as_posix(),
                    "-o",
                    new_file.as_posix(),
                    "-f",
                ],
                # Do not show any output
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print(f"No musicxml found for file {zip_file}, skipping it")
        continue

    for i, file in enumerate(all_files, start=1):
        # Copy the musicxml to the musicxml path
        musicxml_path.mkdir(parents=True, exist_ok=True)
        new_file = (musicxml_path / zip_file.stem).with_suffix(".musicxml")
        if i > 1:
            new_file.with_stem(new_file.stem + f" ({i})")
        # If mxl, unzip it first
        if file.suffix == ".mxl":
            print(file)
            try:
                with zipfile.ZipFile(file) as zipped_xml:
                    elements = zipped_xml.namelist()
                    contained_xml_name = next(
                        filter(lambda f: f.endswith(".xml") or f.endswith(".musicxml"), elements)
                    )
                    with zipped_xml.open(contained_xml_name) as zipped_musicxml:
                        new_file.write_bytes(zipped_musicxml.read())
            except zipfile.BadZipFile:
                if mscz_file := next(file.parent.rglob("*.mscz"), None):
                    print(f"Error trying to unzip {file.as_posix()}, trying with the mscz file")
                    process = subprocess.run(
                        [
                            MUSESCORE_PATH,
                            mscz_file.as_posix(),
                            "-o",
                            new_file.as_posix(),
                            "-f",
                        ]
                    )
                else:
                    print(f"Error trying to unzip {file.as_posix()}, skipping it")
        else:
            shutil.copy(file, new_file)
