#!/bin/bash
# save as create_adr.sh

ADR_DIR="docs/adr"
mkdir -p $ADR_DIR
INDEX=$(ls $ADR_DIR | grep -E '^\d{4}-.*\.md$' | wc -l | xargs)
INDEX=$(printf "%04d" $((INDEX + 1)))

TITLE=$1
FILENAME="$INDEX-$(echo $TITLE | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd '[:alnum:]-').md"

cat <<EOF > "$ADR_DIR/$FILENAME"
# ${INDEX}: $TITLE

Date: $(date +%Y-%m-%d)

## Status
Proposed

## Context
...

## Decision
...

## Alternatives Considered
...

## Consequences
...
EOF

code "$ADR_DIR/$FILENAME"
