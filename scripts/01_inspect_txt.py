from pathlib import Path
import csv

file_path = Path("data/raw/chicago_crime.txt")

print("=" * 80)
print("CHICAGO CRIME TXT FILE INSPECTION")
print("=" * 80)

if not file_path.exists():
    print("ERROR: File not found.")
    print("Expected location:", file_path)
    exit()

file_size_gb = file_path.stat().st_size / (1024 ** 3)
print(f"File found: {file_path}")
print(f"File size: {file_size_gb:.2f} GB")

print("\nReading first 5 lines...\n")

sample_lines = []

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for i in range(5):
        line = f.readline()
        sample_lines.append(line)
        print(f"LINE {i + 1}:")
        print(line[:1000])
        print("-" * 80)

sample_text = "".join(sample_lines)

print("\nDelimiter check:")

possible_delimiters = {
    "comma ,": sample_text.count(","),
    "tab \\t": sample_text.count("\t"),
    "pipe |": sample_text.count("|"),
    "semicolon ;": sample_text.count(";"),
}

for delimiter, count in possible_delimiters.items():
    print(f"{delimiter}: {count}")

print("\nCSV Sniffer result:")

try:
    dialect = csv.Sniffer().sniff(sample_text)
    print("Detected delimiter:", repr(dialect.delimiter))
except Exception as e:
    print("Could not automatically detect delimiter.")
    print("Reason:", e)

print("\nHeader guess:")

first_line = sample_lines[0]

for delimiter in [",", "\t", "|", ";"]:
    columns = first_line.strip().split(delimiter)
    print(f"Using delimiter {repr(delimiter)} gives {len(columns)} columns")
    if len(columns) > 1:
        print(columns[:20])
