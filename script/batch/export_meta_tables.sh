#!/bin/bash
# Exports Meta's tables for the batch matching (match_bulk.py) from a QLever endpoint: CSV, then Parquet
# (DuckDB). Same patterns as the tool's queries and the QLever views, so the same data.
#   bash export_meta_tables.sh test             5 rows a table, to check the format
#   bash export_meta_tables.sh all              all the tables, one at a time
#   bash export_meta_tables.sh <table>          only one: authors titles dates volumes pages doi
#   bash export_meta_tables.sh convert <table>  only the conversion to Parquet, from the CSV already exported
# Settings (environment): QLEVER (endpoint, default http://localhost:7011), QLEVER_TOKEN (access token, needed
# for the long timeout), OUT (output folder, default ./meta_tables).
# Run it when nothing else uses the endpoint: each export reads hundreds of millions of rows.
Q=${QLEVER:-http://localhost:7011}
TOKEN=${QLEVER_TOKEN:-}
OUT=${OUT:-./meta_tables}
P='PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX pro: <http://purl.org/spar/pro/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX fabio: <http://purl.org/spar/fabio/>
PREFIX frbr: <http://purl.org/vocab/frbr/core#>
PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
PREFIX datacite: <http://purl.org/spar/datacite/>
PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>'
declare -A QUERY=(
  # surname of anyone with a role in the work (like the "authors" view), with the role:
  # year_volume_page in the tool asks for pro:withRole pro:author
  [authors]='SELECT ?name ?br ?role WHERE { ?a foaf:familyName ?name . ?r pro:isHeldBy ?a . ?br pro:isDocumentContextFor ?r . OPTIONAL { ?r pro:withRole ?role } }'
  [titles]='SELECT ?br ?title WHERE { ?br dcterms:title ?title }'
  [dates]='SELECT ?br ?date WHERE { ?br prism:publicationDate ?d . BIND(STR(?d) AS ?date) }'
  [volumes]='SELECT ?vol ?br WHERE { ?volume fabio:hasSequenceIdentifier ?vol . ?issue frbr:partOf ?volume . ?br frbr:partOf ?issue . }'
  [pages]='SELECT ?br ?start ?end WHERE { ?br frbr:embodiment ?e . OPTIONAL { ?e prism:startingPage ?start } OPTIONAL { ?e prism:endingPage ?end } }'
  # for the DOI queries (DOI found by GROBID in the text) and the 15 points of the DOI
  [doi]='SELECT ?br ?doi WHERE { ?br datacite:hasIdentifier ?id . ?id datacite:usesIdentifierScheme datacite:doi ; literal:hasLiteralValue ?doi . }'
)
TABLES="authors titles dates volumes pages doi"

export_table() {  # $1 = table, $2 = query suffix (LIMIT for the test), $3 = file suffix
  mkdir -p "$OUT"
  local csv="$OUT/$1$3.csv"
  echo "== $1"
  time curl -sS -f "$Q" -H 'Accept: text/csv' \
    ${TOKEN:+--data-urlencode "access-token=$TOKEN"} --data-urlencode "timeout=86400s" \
    --data-urlencode "query=$P
${QUERY[$1]} $2" -o "$csv" || { echo "ERROR on $1"; return 1; }
  ls -lh "$csv"
}

parquet() {  # $1 = table: CSV -> Parquet, counts the rows
  uv run --no-project --with duckdb python - "$OUT/$1.csv" "$OUT/$1.parquet" <<'EOF'
import duckdb, sys
con = duckdb.connect()
con.sql("SET memory_limit='20GB'; SET threads=8")
# quotes declared: the automatic detection looks at the first rows and, if there are none there, it then
# rejects a field like "16,32,58" (pages)
con.sql(f"COPY (SELECT * FROM read_csv('{sys.argv[1]}', header=true, all_varchar=true, delim=',', "
        f"quote='\"', escape='\"')) "
        f"TO '{sys.argv[2]}' (FORMAT parquet, COMPRESSION zstd)")
print(sys.argv[2], con.sql(f"SELECT count(*) FROM '{sys.argv[2]}'").fetchone()[0], 'rows')
EOF
}

case "$1" in
  test) for t in $TABLES; do export_table "$t" "LIMIT 5" _test && cat "$OUT/${t}_test.csv"; done ;;
  all) for t in $TABLES; do export_table "$t" && parquet "$t"; done; du -sh "$OUT"/* ;;
  authors|titles|dates|volumes|pages|doi) export_table "$1" && parquet "$1" ;;
  convert) parquet "$2" ;;   # only CSV -> Parquet, from the CSV already exported
  *) echo "usage: bash export_meta_tables.sh test|all|authors|titles|dates|volumes|pages|doi|convert <table>" ;;
esac
