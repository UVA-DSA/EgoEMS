#!/usr/bin/env bash
# EgoEMS downloader — Harvard Dataverse DOI:10.7910/DVN/XT51K7
# Requirements: bash 4+, curl, jq. Optional: md5sum/shasum for checksum verification.
set -uo pipefail

BASE_URL="https://dataverse.harvard.edu"
DATASET_DOI="doi:10.7910/DVN/XT51K7"
VERSION=":latest-published"
CONCURRENCY=4
OUTPUT_DIR="./EgoEMS"
requested_files=''
NO_COLOR=${NO_COLOR:-}

ansi_terminal=false
if [[ -t 1 && ${TERM:-dumb} != dumb ]]; then ansi_terminal=true; fi

if [[ "$ansi_terminal" == true && -z "$NO_COLOR" ]]; then
  BLUE=$'\033[38;5;39m'; GREEN=$'\033[38;5;42m'
  YELLOW=$'\033[38;5;214m'; RED=$'\033[38;5;196m'
  DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
  BLUE=''; GREEN=''; YELLOW=''; RED=''; DIM=''; BOLD=''; RESET=''
fi

say() { printf '%s\n' "$*"; }
info() { printf '%s>%s %s\n' "$BLUE" "$RESET" "$*"; }
good() { printf '%s+%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%s!%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
fail() { printf '%s!%s %s\n' "$RED" "$RESET" "$*" >&2; }
die() { fail "$*"; exit 1; }

usage() {
  say "Usage: bash download_egoems.sh [--output DIR] [--jobs N] [--files LIST] [--help]"
  say ""
  say "  --output DIR   Save dataset here (default: ./EgoEMS)"
  say "  --jobs N       Concurrent files, 1-8 (default: 4)"
  say "  --files LIST   ZIP names, e.g. cars_10,cars_4 (default: all)"
  say "  --help         Show this help"
  say ""
  say "Interactive runs prompt for ZIP names if --files is omitted."
  say "Set EGOEMS_NO_ANIMATION=1 for a static display."
}

while (($#)); do
  case "$1" in
    -o|--output) (($# >= 2)) || die "Missing output directory"; OUTPUT_DIR=$2; shift 2 ;;
    -j|--jobs) (($# >= 2)) || die "Missing job count"; CONCURRENCY=$2; shift 2 ;;
    --files)
      [[ $# -ge 2 && -n "$2" ]] || die "Give --files one or more ZIP names, e.g. cars_10,cars_4."
      requested_files=$2
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (try --help)" ;;
  esac
done
[[ "$CONCURRENCY" =~ ^[1-8]$ ]] || die "--jobs must be a number from 1 to 8"
for dep in bash curl jq; do
  command -v "$dep" >/dev/null 2>&1 || die "Missing $dep. On Ubuntu/WSL: sudo apt install curl jq"
done

terminal_size=''
show_banner() {
  local columns subtitle='Harvard Dataverse - Dataset Downloader'
  if [[ "$ansi_terminal" == true ]]; then
    terminal_size=$(stty size <&3 2>/dev/null) || terminal_size=''
  fi
  read -r _ columns <<< "$terminal_size"

  say ""
  if [[ "$columns" =~ ^[0-9]+$ ]] && (( columns >= 54 )); then
    printf '%s%s' "$BLUE" "$BOLD"
    printf '%s\n' \
      '  ######   ####    ####   ######  ##   ##   #####' \
      '  ##      ##      ##  ##  ##      ### ###  ##' \
      '  #####   ## ###  ##  ##  #####   ## # ##   ####' \
      '  ##      ##  ##  ##  ##  ##      ##   ##      ##' \
      '  ######   ####    ####   ######  ##   ##  #####'
    printf '%s\n' "$RESET${BOLD}      $subtitle$RESET"
  else
    say "${BLUE}${BOLD}EgoEMS${RESET}"
    if [[ "$columns" =~ ^[0-9]+$ ]] && (( columns < ${#subtitle} + 1 )); then
      say 'Harvard Dataverse'
      say 'Dataset Downloader'
    else
      say "$subtitle"
    fi
  fi
  say ""
}
show_banner 3<&1

if [[ -t 0 && "$OUTPUT_DIR" == './EgoEMS' ]]; then
  read -r -p "Download directory [./EgoEMS]: " entered_dir
  [[ -z "$entered_dir" ]] || OUTPUT_DIR=$entered_dir
fi

[[ -n "$OUTPUT_DIR" ]] || die "Output directory cannot be empty"
mkdir -p -- "$OUTPUT_DIR" || die "Cannot create: $OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd -P)

say ""
info "Output: $OUTPUT_DIR"
info "Parallel downloads: $CONCURRENCY"

cache_home=${XDG_CACHE_HOME:-${HOME:+$HOME/.cache}}
cache_dir=${cache_home:+$cache_home/egoems}
cache_file=${cache_dir:+$cache_dir/download_egoems.json}
cached_profile=''
if [[ -n "$cache_file" && -f "$cache_file" ]]; then
  info "To update saved details, delete $cache_file and rerun."
  if [[ -L "$cache_file" || ! -O "$cache_file" ]] ||
    ! cached_profile=$(jq -cse 'select(length == 1) | first | objects
      | select((.token | type) == "string" and (.guestbookResponse | type) == "object")' "$cache_file" 2>/dev/null); then
    warn "Could not read saved details. Enter them again for this run."
    cached_profile=''
  fi
fi

TOKEN=${DATAVERSE_TOKEN:-}
if [[ -t 0 && -n "$TOKEN" ]]; then
  read -r -p "Use DATAVERSE_TOKEN from this terminal? [Y/n]: " use_token || die "Could not read token preference."
  if [[ "$use_token" =~ ^[Nn] ]]; then TOKEN=''; fi
elif [[ -z "$TOKEN" && -n "$cached_profile" ]]; then
  TOKEN=$(jq -r '.token | strings' <<< "$cached_profile")
fi
if [[ -t 0 && -z "$TOKEN" ]]; then
  say "${DIM}An API token with download access is required for restricted files.${RESET}"
  say "${DIM}Press Enter to continue with public files only.${RESET}"
  read -r -s -p "Dataverse API token: " TOKEN || die "Could not read the API token."
  say ""
fi

headers=()
has_token=false
if [[ -n "$TOKEN" ]]; then
  has_token=true
  headers=(-H "X-Dataverse-key: $TOKEN")
fi

version_url="$BASE_URL/api/datasets/:persistentId/versions/$VERSION"
info "Reading published dataset metadata..."
metadata=$(curl -fsSL --connect-timeout 20 --max-time 60 --retry 2 \
  "${headers[@]}" -G "$version_url" \
  --data-urlencode "persistentId=$DATASET_DOI" \
  --data-urlencode "excludeFiles=false") || die "Could not retrieve dataset metadata. Check your connection and API token."
[[ $(jq -r '.status // ""' <<< "$metadata") == OK ]] || die "Dataverse returned an error: $(jq -r '.message // "Unknown error"' <<< "$metadata")"

read -r published_count restricted_count < <(jq -r '(.data.files // []) | [length,
  (map(select(.restricted == true or .dataFile.restricted == true)) | length)] | @tsv' <<< "$metadata") \
  || die "Could not read the file list in the Dataverse response."
(( published_count > 0 )) || die "No files were found in the published dataset."

if (( restricted_count == published_count )); then
  [[ "$has_token" == true ]] || die "All $restricted_count files are restricted. Run again with a Harvard Dataverse API token from an account with download access."
  permissions=$(curl -fsSG --connect-timeout 20 --max-time 60 --retry 2 \
    "${headers[@]}" "$version_url/canDownloadAtLeastOneFile" \
    --data-urlencode "persistentId=$DATASET_DOI") || die "Could not check download permissions. Check your connection and API token."
  case $(jq -r 'if .status == "OK" and (.data | type) == "boolean" then .data else empty end' <<< "$permissions") in
    true) ;;
    false) die "This API token's account cannot download any of the $restricted_count restricted files. Request access in Harvard Dataverse." ;;
    *) die "Dataverse returned an unexpected download-permission response." ;;
  esac
elif (( restricted_count > 0 )) && [[ "$has_token" == false ]]; then
  warn "Skipping $restricted_count restricted files. Use an API token from an account with download access to include them."
fi

select_files() {
  jq -r --arg requested "$1" --argjson has_token "$has_token" '
    def accessible: $has_token or (.restricted != true and .dataFile.restricted != true);
    (.data.files // []) as $files
    | if $requested == "" then
        $files | map(select(accessible))
      else
        ($requested | split(",") | map(gsub("^\\s+|\\s+$"; ""))) as $names
        | reduce $names[] as $name ([];
            if $name == "" then error("Enter a ZIP name for each comma-separated entry.") else . end
            | ($name | if endswith(".zip") then . else . + ".zip" end) as $label
            | [$files[] | select(.label == $label)] as $matches
            | if ($matches | length) == 0 then error("No published ZIP file named \($label).")
              elif ($matches | length) > 1 then error("More than one published file is named \($label).")
              elif ($matches[0] | accessible | not) then error("Download access and an API token are required for \($label).")
              elif any(.[]; .label == $label) then .
              else . + $matches
              end)
      end
    | .[] | @base64' <<< "$metadata"
}

if [[ -t 0 && -z "$requested_files" ]]; then
  while :; do
    read -r -p "ZIP names [Enter for all] (e.g. cars_10,cars_4): " requested_files || die "Could not read file selection."
    if selected_records=$(select_files "$requested_files"); then break; fi
  done
else
  selected_records=$(select_files "$requested_files") || die "Invalid file selection."
fi
[[ -n "$selected_records" ]] || die "No accessible files were found in the published dataset."
mapfile -t FILE_RECORDS <<< "$selected_records"
TOTAL=${#FILE_RECORDS[@]}
say "${GREEN}${BOLD}Selected $TOTAL of $published_count published files.${RESET}"

gb_name=${EGOEMS_NAME:-}
gb_email=${EGOEMS_EMAIL:-}
gb_institution=${EGOEMS_INSTITUTION:-}
gb_position=${EGOEMS_POSITION:-}
if [[ -n "$cached_profile" ]]; then
  gb_name=${gb_name:-$(jq -r '.guestbookResponse.name | strings' <<< "$cached_profile")}
  gb_email=${gb_email:-$(jq -r '.guestbookResponse.email | strings' <<< "$cached_profile")}
  gb_institution=${gb_institution:-$(jq -r '.guestbookResponse.institution | strings' <<< "$cached_profile")}
  gb_position=${gb_position:-$(jq -r '.guestbookResponse.position | strings' <<< "$cached_profile")}
fi
if [[ -z "$gb_name" || -z "$gb_email" || -z "$gb_institution" || -z "$gb_position" ]]; then
  [[ -t 0 ]] || die "Set EGOEMS_NAME, EGOEMS_EMAIL, EGOEMS_INSTITUTION, and EGOEMS_POSITION or save them in an interactive run."
  say ""
  say "${BOLD}Guestbook${RESET} ${DIM}(required by EgoEMS for every download)${RESET}"
  [[ -n "$gb_name" ]] || read -r -p "Full name: " gb_name || die "Could not read guestbook name."
  [[ -n "$gb_email" ]] || read -r -p "Email: " gb_email || die "Could not read guestbook email."
  [[ -n "$gb_institution" ]] || read -r -p "Institution: " gb_institution || die "Could not read guestbook institution."
  [[ -n "$gb_position" ]] || read -r -p "Position / role: " gb_position || die "Could not read guestbook position."
fi
[[ -n "$gb_name" && -n "$gb_email" && -n "$gb_institution" && -n "$gb_position" ]] || die "All four guestbook fields are required."
[[ "$gb_email" == *@*.* ]] || die "Please provide a valid email address."

guestbook_json=$(jq -nc --arg name "$gb_name" --arg email "$gb_email" \
  --arg institution "$gb_institution" --arg position "$gb_position" \
  '{guestbookResponse:{name:$name,email:$email,institution:$institution,position:$position,answers:[]}}') \
  || die "Could not prepare the guestbook response."

if [[ -t 0 ]]; then
  say ""
  file_word=files
  if (( TOTAL == 1 )); then file_word=file; fi
  read -r -p "Start downloading $TOTAL $file_word? [Y/n]: " confirm || die "Could not read download confirmation."
  [[ ! "$confirm" =~ ^[Nn] ]] || { say "Cancelled."; exit 0; }
fi

save_profile() (
  umask 077
  mkdir -p -- "$cache_dir" || return 1
  [[ ! -L "$cache_dir" && -O "$cache_dir" && ! -L "$cache_file" ]] || return 1
  if [[ -e "$cache_file" ]]; then [[ -f "$cache_file" && -O "$cache_file" ]] || return 1; fi
  temporary_profile=$(mktemp "$cache_dir/.download_egoems.XXXXXXXX") || return 1
  trap 'rm -f -- "$temporary_profile"' EXIT
  printf '%s\0%s' "$TOKEN" "$guestbook_json" \
    | jq -Rsc 'split("\u0000") | (.[1] | fromjson) + {token: .[0]}' > "$temporary_profile" || return 1
  mv -f -- "$temporary_profile" "$cache_file"
)

if [[ -n "$cache_file" ]]; then
  if save_profile; then
    if [[ -z "$cached_profile" ]]; then info "To update saved details, delete $cache_file and rerun."; fi
  else
    warn "Could not save token and guestbook details in $cache_file."
  fi
fi

# Temporary status files do not contain signed URLs or API tokens.
status_dir=$(mktemp -d) || die "Cannot create temporary status directory"
animation_pid=''
stop_animation() {
  [[ -n "$animation_pid" ]] || return 0
  if [[ $(jobs -rp) == "$animation_pid" ]]; then
    kill "$animation_pid" 2>/dev/null || :
  fi
  wait "$animation_pid" 2>/dev/null || :
  animation_pid=''
}

cleanup() {
  stop_animation
  rm -f -- "$status_dir"/*.status "$status_dir"/*.log 2>/dev/null || :
  rmdir -- "$status_dir" 2>/dev/null || :
}
trap cleanup EXIT

check_file() {
  local path=$1 expected=$2 algorithm=$3 hash=$4 actual
  [[ -f "$path" ]] || return 1
  [[ $(wc -c < "$path") -eq "$expected" ]] || return 1
  if [[ -n "$hash" && "$algorithm" == MD5 ]] && command -v md5sum >/dev/null; then
    actual=$(md5sum -- "$path"); actual=${actual%% *}
    [[ "${actual,,}" == "${hash,,}" ]] || return 1
  elif [[ -n "$hash" && "$algorithm" == SHA-1 ]] && command -v sha1sum >/dev/null; then
    actual=$(sha1sum -- "$path"); actual=${actual%% *}
    [[ "${actual,,}" == "${hash,,}" ]] || return 1
  fi
  return 0
}

download_one() {
  local n=$1 record=$2 id filename size algorithm hash path response http_code url attempt
  record=$(printf '%s' "$record" | base64 --decode)
  id=$(jq -r '.dataFile.id' <<< "$record")
  filename=$(jq -r '.label' <<< "$record")
  size=$(jq -r '.dataFile.filesize // 0' <<< "$record")
  algorithm=$(jq -r '.dataFile.checksum.type // ""' <<< "$record")
  hash=$(jq -r '.dataFile.checksum.value // ""' <<< "$record")

  # Guard against path traversal and conflicting names.
  if [[ "$filename" == */* || "$filename" == *'..'* || "$filename" == .* || -z "$filename" ]]; then
    fail "Unsafe filename in metadata: $filename"
    : > "$status_dir/$n.failed.status"
    return 1
  fi
  path="$OUTPUT_DIR/$filename"
  if check_file "$path" "$size" "$algorithm" "$hash"; then
    good "[$n/$TOTAL] Already verified: $filename"
    : > "$status_dir/$n.skipped.status"
    return 0
  fi
  for attempt in 1 2 3; do
    info "[$n/$TOTAL] ${filename} - attempt $attempt/3"
    # POST returns a short-lived signed URL. Request immediately before each attempt.
    if ! response=$(curl -sS --connect-timeout 20 --max-time 60 --write-out $'\n%{http_code}' \
      "${headers[@]}" -H 'Content-Type: application/json' \
      -d "$guestbook_json" "$BASE_URL/api/access/datafile/$id"); then
      warn "[$n/$TOTAL] Guestbook request could not reach Dataverse for $filename."
      sleep 3
      continue
    fi
    http_code=${response##*$'\n'}
    case "$http_code" in
      2[0-9][0-9]) ;;
      401|403)
        warn "[$n/$TOTAL] Download authorization denied for $filename (HTTP $http_code). Check your API token and file permissions."
        break
        ;;
      408|429|5[0-9][0-9])
        warn "[$n/$TOTAL] Guestbook request failed for $filename (HTTP $http_code). Retrying."
        sleep 3
        continue
        ;;
      *)
        warn "[$n/$TOTAL] Guestbook request rejected for $filename (HTTP $http_code)."
        break
        ;;
    esac
    if ! url=$(jq -er '.data.signedUrl | strings | select(length > 0)' <<< "${response%$'\n'*}" 2>/dev/null); then
      warn "[$n/$TOTAL] Dataverse returned no signed download URL for $filename (HTTP $http_code)."
      break
    fi
    # A new signed URL on every attempt makes failures/restarts recoverable.
    # --silent avoids accidentally printing signed URLs to shared terminal logs.
    if curl --silent --show-error --fail --location \
      --connect-timeout 30 --speed-time 90 --speed-limit 1024 \
      --continue-at - --output "$path" "$url" 2>/dev/null; then
      if check_file "$path" "$size" "$algorithm" "$hash"; then
        good "[$n/$TOTAL] Verified: $filename"
        : > "$status_dir/$n.downloaded.status"
        return 0
      fi
      warn "[$n/$TOTAL] Size/checksum mismatch: $filename"
      # A checksum mismatch may mean local data corruption; do not append to it.
      mv -n -- "$path" "$path.invalid.$(date +%s).$n" || :
    else
      warn "[$n/$TOTAL] Transfer interrupted: $filename"
    fi
    sleep 3
  done
  fail "[$n/$TOTAL] Could not download: $filename"
  : > "$status_dir/$n.failed.status"
  return 1
}

animate_download() {
  # Digits mark the six arms clockwise from the top. Braces mark the staff and serpent.
  local -a star_mask=(
    '          0000000'
    '          000{|}000'
    '   555    0{o-|}000    111'
    '  5555555 000{|`-.} 1111111'
    ' 555555555500{|  )}111111111'
    '    555555{ .-+-/}1111111'
    '       444{(  |}111111'
    '    4444444{`-+-.}2222222'
    ' 444444444433{|  )}222222222'
    '  4444444 3{ /+-/}3 2222222'
    '   444    333{|}333    222'
    '          333{|}333'
    '          3333333'
  )
  local -a compact_pulse=(' * ' '.*.' '-*-' '=*=' '-*-' '.*.')
  local art_width=26 stacked_rows=$((${#star_mask[@]} + 1)) framed_rows=$((${#star_mask[@]} + 2))
  local phase=0 size='' previous_size='' rows columns panel_rows=0 padding
  local verified failed percent detail failure filled empty bar progress trail row line visible painted
  local next_row=$'\033[K\033[E'
  local outline='  +----------------------------+-----------------------------------+'
  local edge="$DIM$BLUE|$RESET" pulse_color='' trail_color=''
  local -a details
  if [[ -n "$BLUE" ]]; then
    pulse_color=$'\033[38;5;45m'
    trail_color=$'\033[38;5;75m'
  fi
  shopt -s nullglob
  trap 'exit 0' TERM
  trap 'if (( panel_rows > 0 )); then printf "\r\033[J"; fi' EXIT

  while :; do
    size=$(stty size <&3 2>/dev/null) || size=$previous_size
    if [[ "$size" != "$previous_size" ]]; then
      # Reserve the animation area again after a resize. The cursor rests at its top.
      if (( panel_rows > 0 )); then printf '\r\033[J'; fi
      panel_rows=0
      read -r rows columns <<< "$size"
      if [[ "$rows" =~ ^[0-9]+$ && "$columns" =~ ^[0-9]+$ ]] && (( rows >= 2 && columns >= 12 )); then
        panel_rows=1
        if (( rows >= framed_rows + 2 && columns > ${#outline} )); then
          panel_rows=$framed_rows
        elif (( rows >= stacked_rows + 2 && columns >= art_width + 2 )); then
          panel_rows=$stacked_rows
        fi
        if (( panel_rows > 1 )); then
          printf -v padding '%*s' "$((panel_rows - 1))" ''
          printf '%s\033[%dA\r' "${padding// /$'\n'}" "$((panel_rows - 1))"
        fi
      fi
      previous_size=$size
    fi

    if (( panel_rows > 0 )); then
      set -- "$status_dir"/*.downloaded.status "$status_dir"/*.skipped.status
      verified=$#
      set -- "$status_dir"/*.failed.status
      failed=$#
      failure=''
      if (( failed > 0 )); then failure="$failed failed"; fi
      printf -v percent '%3d%%' "$((verified * 100 / TOTAL))"
      if (( panel_rows == 1 )); then
        progress="${compact_pulse[phase]} $percent  $verified/$TOTAL ok"
        if (( failed > 0 )); then progress="$failure  $verified/$TOTAL ok"; fi
        printf '\r%s\033[K' "${progress:0:columns-1}"
      else
        detail="$verified/$TOTAL files verified"
        printf -v filled '%*s' "$((verified * 20 / TOTAL))" ''
        printf -v empty '%*s' "$((20 - verified * 20 / TOTAL))" ''
        bar="[${filled// /#}${empty// /.}]"
        painted=''
        if (( panel_rows == framed_rows )); then
          details=('' '' '' '' '' 'Downloading dataset' "$bar  $percent" "$detail" "$failure" '' '' '' '')
          painted="$DIM$BLUE$outline$RESET$next_row"
        fi

        trail=$(((phase + 5) % 6))
        for row in "${!star_mask[@]}"; do
          line=${star_mask[row]//"$phase"/#}
          line=${line//"$trail"/=}
          line=${line//[0-5]/:}
          visible=${line//[\{\}]/}
          line=${line//\#/"$pulse_color#$BLUE"}
          line=${line//=/"$trail_color=$BLUE"}
          line=${line//\{/"$RESET$BOLD"}
          line=${line//\}/"$RESET$BLUE"}
          line="$BLUE$line$RESET"
          if (( panel_rows == framed_rows )); then
            printf -v padding '%*s' "$((art_width - ${#visible}))" ''
            printf -v line '  %s %s%s %s  %-31.31s  %s' \
              "$edge" "$line" "$padding" "$edge" "${details[row]}" "$edge"
          fi
          painted+="$line$next_row"
        done

        if (( panel_rows == framed_rows )); then
          painted+="$DIM$BLUE$outline$RESET"
        else
          progress="$percent  $detail"
          if (( failed > 0 )); then progress="$failure  $verified/$TOTAL verified"; fi
          if (( columns > ${#bar} + ${#progress} + 2 )); then
            progress="$bar  $progress"
          fi
          painted+=${progress:0:columns-1}
        fi
        printf '\r\0337%s\033[K\0338' "$painted"
      fi
      phase=$(((phase + 1) % 6))
    fi
    sleep 0.16 || exit 0
  done
}

say ""
if [[ -n "$terminal_size" && -z "${EGOEMS_NO_ANIMATION:-}" ]]; then
  animate_download 3<&1 &
  animation_pid=$!
else
  say "${BLUE}${BOLD}-- Downloading --${RESET}"
fi

# Keep the renderer out of the transfer job table so refreshes never gate the queue.
(
  for i in "${!FILE_RECORDS[@]}"; do
    while mapfile -t active < <(jobs -rp) && [[ -n ${active[0]:-} ]] && (( ${#active[@]} >= CONCURRENCY )); do
      if (( BASH_VERSINFO[0] > 4 || BASH_VERSINFO[1] >= 3 )); then
        wait -n || :
      else
        wait "${active[0]}" || :
      fi
    done
    download_one "$((i+1))" "${FILE_RECORDS[i]}" >"$status_dir/$((i+1)).log" 2>&1 &
  done
  wait || :
)
stop_animation

downloaded=0; skipped=0; failed=0
for ((i=1; i<=TOTAL; i++)); do
  if [[ -f "$status_dir/$i.downloaded.status" ]]; then
    ((downloaded+=1))
  elif [[ -f "$status_dir/$i.skipped.status" ]]; then
    ((skipped+=1))
  else
    ((failed+=1))
  fi
done
say ""
say "${BOLD}-- Download Summary --${RESET}"
printf '  Downloaded: %s%s%s\n' "$GREEN" "$downloaded" "$RESET"
printf '  Already present: %s\n' "$skipped"
printf '  Failed: %s%s%s\n' "$RED" "$failed" "$RESET"
say "  Folder: $OUTPUT_DIR"
if (( failed > 0 )); then
  say "${BOLD}Failed transfers:${RESET}"
  for ((i=1; i<=TOTAL; i++)); do
    if [[ -f "$status_dir/$i.failed.status" ]]; then
      tail -n 2 -- "$status_dir/$i.log" >&2
    fi
  done
  warn "Resolve the errors above before rerunning. Verified files are skipped on retry."
  exit 1
fi
good "All $TOTAL files are present and verified!"
