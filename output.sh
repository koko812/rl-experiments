#!/bin/bash

OUTPUT="$HOME/Downloads/combined_output.txt"
> "$OUTPUT"  # 出力ファイルを空にする

# 対象拡張子
EXTENSIONS="py md"

# 対象ディレクトリ（.venv 除外）
find . -type f \( -name "*.py" -o -name "*.md" \) ! -path "./.venv/*" ! -path "./.git/*" | sort | while read file; do
  echo -e "\n\n# ==== $file ====" >> "$OUTPUT"
  cat "$file" >> "$OUTPUT"
done

echo "✅ 結合完了: $OUTPUT"

