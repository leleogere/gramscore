#!/bin/bash
# This script is adapted from the Nakamura's MIDIToMIDIAlign.sh script,
# but with more flexibility regarding the working directory.

set -e

if [ $# -ne 4 ]; then
    echo "Usage: $0 <reference.mid> <other.mid> <base_path> <nakamura_path>"
    echo "<reference.mid> and <other.mid> must end with .mid and belong to <base_path>"
    exit 1
fi

reference="${1%.*}"
other="${2%.*}"
base_path="$3"
nakamura_path="$4"
program_folder="$nakamura_path/Programs"

# Convert MIDI files to pianoroll
"$program_folder/midi2pianoroll" 0 "$base_path/$reference"
"$program_folder/midi2pianoroll" 0 "$base_path/$other"

# Process reference MIDI
"$program_folder/SprToFmt3x" "$base_path/${reference}_spr.txt" "$base_path/${reference}_fmt3x.txt"
"$program_folder/Fmt3xToHmm" "$base_path/${reference}_fmt3x.txt" "$base_path/${reference}_hmm.txt"

# Perform alignment
"$program_folder/ScorePerfmMatcher" "$base_path/${reference}_hmm.txt" "$base_path/${other}_spr.txt" "$base_path/${other}_pre_match.txt" 0.001
"$program_folder/ErrorDetection" "$base_path/${reference}_fmt3x.txt" "$base_path/${reference}_hmm.txt" "$base_path/${other}_pre_match.txt" "$base_path/${other}_err_match.txt" 0
"$program_folder/RealignmentMOHMM" "$base_path/${reference}_fmt3x.txt" "$base_path/${reference}_hmm.txt" "$base_path/${other}_err_match.txt" "$base_path/${other}_realigned_match.txt" 0.3

# Finalize output
mv "$base_path/${other}_realigned_match.txt" "$base_path/${other}_match.txt"
"$program_folder/MatchToCorresp" "$base_path/${other}_match.txt" "$base_path/${reference}_spr.txt" "$base_path/${other}_corresp.txt"

# Cleanup
rm -f "$base_path/${other}_err_match.txt" "$base_path/${other}_pre_match.txt"
