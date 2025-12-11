import json
from pathlib import Path

LOG_DIR = Path("logs")

def convert_log_to_json(log_file: Path):
    json_file = log_file.with_suffix(".json")
    data = []

    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Basic parsing: "[timestamp] LEVEL: message"
            try:
                timestamp_end = line.index("]") + 1
                timestamp = line[1:timestamp_end-1]
                rest = line[timestamp_end+1:].strip()
                level_end = rest.index(":")
                level = rest[:level_end].strip()
                message = rest[level_end+1:].strip()
            except Exception as e:
                # fallback for unparseable lines
                timestamp, level, message = None, None, line

            data.append({
                "timestamp": timestamp,
                "level": level,
                "message": message
            })

    with json_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Converted {log_file} -> {json_file}")

def main():
    for log_file in LOG_DIR.rglob("*.log"):
        convert_log_to_json(log_file)

if __name__ == "__main__":
    main()
