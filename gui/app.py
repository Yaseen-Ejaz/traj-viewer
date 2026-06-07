from __future__ import annotations

import json
import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REQUIRED_FIELDS_JSONL = {"instance_id", "file_name", "file_str"}


st.set_page_config(
    page_title="Trajectory Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── helpers ──────────────────────────────────────────────────────────────────

def render_wrapped_content(content: str) -> None:
    st.html(
        f"""
        <pre style="
            background-color: rgba(128, 128, 128, 0.12);
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.5rem;
            color: inherit;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                'Liberation Mono', 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
            max-width: 100%;
            overflow-x: hidden;
            overflow-wrap: anywhere;
            padding: 1rem;
            white-space: pre-wrap;
            word-break: break-word;
        ">{escape(content)}</pre>
        """
    )


def display_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def try_parse_json(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def clamp_index(value: Any, size: int) -> int:
    if size <= 0:
        return 0
    try:
        index = int(value)
    except (TypeError, ValueError):
        index = 0
    return max(0, min(index, size - 1))


def unique_options(series: pd.Series) -> list[str]:
    values = [str(v) for v in series.dropna().unique() if str(v) != ""]
    return sorted(values)


def safe_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def normalize_boolish(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value in ("True", "False"):
        return str(value)
    return ""


# ── navigation helpers ────────────────────────────────────────────────────────

def sync_selected_label(offset_key: str, label_key: str, labels: list[str]) -> int:
    offset = clamp_index(st.session_state.get(offset_key, 0), len(labels))
    current_label = st.session_state.get(label_key)
    if current_label in labels:
        offset = labels.index(current_label)
    else:
        st.session_state[label_key] = labels[offset]
    st.session_state[offset_key] = offset
    return offset


def render_navigation_buttons(
    offset_key: str,
    label_key: str,
    labels: list[str],
    key_prefix: str,
) -> int:
    offset = sync_selected_label(offset_key, label_key, labels)
    back_col, next_col, spacer_col = st.columns([1, 1, 6])
    if back_col.button("Back", key=f"{key_prefix}_back", disabled=offset == 0):
        offset = clamp_index(offset - 1, len(labels))
        st.session_state[offset_key] = offset
        st.session_state[label_key] = labels[offset]
    if next_col.button("Next", key=f"{key_prefix}_next", disabled=offset >= len(labels) - 1):
        offset = clamp_index(offset + 1, len(labels))
        st.session_state[offset_key] = offset
        st.session_state[label_key] = labels[offset]
    spacer_col.caption(f"{offset + 1} of {len(labels)}")
    return offset


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE A — JSONL trajectory viewer (existing functionality)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_jsonl_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.glob("*.jsonl") if path.is_file())


def is_compatible_jsonl(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                return isinstance(record, dict) and REQUIRED_FIELDS_JSONL.issubset(record)
    except (OSError, json.JSONDecodeError):
        return False
    return False


@st.cache_data(show_spinner=False)
def load_jsonl(path_str: str, mtime_ns: int) -> tuple[list[dict[str, Any]], list[str]]:
    del mtime_ns
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"Line {line_number}: expected a JSON object")
                continue
            record["_line_number"] = line_number
            rows.append(record)
    return rows, errors


def summarize_records(rows: list[dict[str, Any]]) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        interaction_count = safe_len(row.get("interaction_str"))
        summary_rows.append({
            "_row_index": index,
            "line": row.get("_line_number", index + 1),
            "edit_type": row.get("edit_type", ""),
            "instance_id": row.get("instance_id", ""),
            "file_name": row.get("file_name", ""),
            "file_classification": row.get("file_classification", ""),
            "test_label": row.get("test_class_deterministic_label", ""),
            "needs_review": normalize_boolish(row.get("test_class_needs_review", "")),
            "churn": int(row.get("churn") or 0),
            "additions": int(row.get("additions") or 0),
            "deletions": int(row.get("deletions") or 0),
            "interactions": interaction_count,
        })
    return pd.DataFrame(summary_rows)


def contains_query(row: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    haystack = "\n".join([
        display_scalar(row.get("instance_id", "")),
        display_scalar(row.get("edit_type", "")),
        display_scalar(row.get("file_name", "")),
        display_scalar(row.get("file_classification", "")),
        display_scalar(row.get("test_class_deterministic_label", "")),
        display_scalar(row.get("test_class_suggested_label", "")),
        display_scalar(row.get("test_class_review_reason", "")),
        display_scalar(row.get("file_str", "")),
        display_scalar(row.get("interaction_numbers", "")),
        display_scalar(row.get("interaction_str", "")),
    ]).lower()
    return query.lower() in haystack


def apply_jsonl_filters(summary: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        query = st.text_input("Search", placeholder="instance, file, diff, interaction")
        edit_type_options = unique_options(summary["edit_type"])
        selected_edit_types = st.multiselect("Edit type", edit_type_options)
        class_options = unique_options(summary["file_classification"])
        selected_classes = st.multiselect("File classification", class_options)
        label_options = unique_options(summary["test_label"])
        selected_labels = st.multiselect("Test label", label_options)
        review_options = unique_options(summary["needs_review"])
        selected_review = st.multiselect("Needs review", review_options)
        only_with_interactions = st.checkbox("Only rows with interactions")

    filtered = summary.copy()
    if selected_edit_types:
        filtered = filtered[filtered["edit_type"].isin(selected_edit_types)]
    if selected_classes:
        filtered = filtered[filtered["file_classification"].isin(selected_classes)]
    if selected_labels:
        filtered = filtered[filtered["test_label"].isin(selected_labels)]
    if selected_review:
        filtered = filtered[filtered["needs_review"].isin(selected_review)]
    if only_with_interactions:
        filtered = filtered[filtered["interactions"] > 0]
    if query:
        matching_indexes = {
            index for index, row in enumerate(rows) if contains_query(row, query)
        }
        filtered = filtered[filtered["_row_index"].isin(matching_indexes)]
    return filtered


def render_jsonl_metrics(summary: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    total_interactions = int(summary["interactions"].sum()) if not summary.empty else 0
    total_churn = int(summary["churn"].sum()) if not summary.empty else 0
    instances = summary["instance_id"].nunique() if not summary.empty else 0
    files = summary["file_name"].nunique() if not summary.empty else 0
    cols = st.columns(5)
    cols[0].metric("Records", f"{len(rows):,}")
    cols[1].metric("Instances", f"{instances:,}")
    cols[2].metric("Files", f"{files:,}")
    cols[3].metric("Total churn", f"{total_churn:,}")
    cols[4].metric("Interactions", f"{total_interactions:,}")


def dataframe_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "line", "edit_type", "instance_id", "file_name", "file_classification",
        "test_label", "needs_review", "churn", "additions", "deletions", "interactions",
    ]
    return frame[columns].reset_index(drop=True)


def render_jsonl_metadata(row: dict[str, Any]) -> None:
    metadata_fields = [
        "edit_type", "agent_type", "agent_run_id", "instance_id", "file_name",
        "file_classification", "churn", "additions", "deletions",
        "test_class_deterministic_label", "test_class_suggested_label",
        "test_class_confidence", "test_class_needs_review", "test_class_review_reason",
    ]
    metadata = {field: row.get(field, "") for field in metadata_fields if field in row}
    st.json(metadata, expanded=False)
    signals = try_parse_json(row.get("test_class_signals", ""))
    if signals:
        st.subheader("Test Signals")
        st.json(signals, expanded=False)


def parse_interaction(raw_interaction: Any) -> dict[str, Any]:
    parsed = try_parse_json(raw_interaction)
    return parsed if isinstance(parsed, dict) else {"content": display_scalar(parsed)}


def interaction_label(number: Any, interaction: dict[str, Any]) -> str:
    role = interaction.get("role", "unknown")
    content = display_scalar(interaction.get("content", "")).replace("\n", " ")
    snippet = content[:80] + ("..." if len(content) > 80 else "")
    return f"{number} | {role} | {snippet}" if snippet else f"{number} | {role}"


def render_tool_calls(tool_calls: Any) -> None:
    if not isinstance(tool_calls, list) or not tool_calls:
        return
    st.subheader("Tool Calls")
    for index, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            st.json(tool_call, expanded=True)
            continue
        function = tool_call.get("function")
        function = function if isinstance(function, dict) else {}
        name = display_scalar(function.get("name", ""))
        title = f"Tool Call {index}" + (f": {name}" if name else "")
        st.markdown(f"**{title}**")
        id_col, type_col, name_col = st.columns(3)
        id_col.metric("ID", display_scalar(tool_call.get("id", "")))
        type_col.metric("Type", display_scalar(tool_call.get("type", "")))
        name_col.metric("Function", name)
        arguments = function.get("arguments", "")
        if arguments:
            parsed_arguments = try_parse_json(arguments)
            st.caption("Arguments")
            if isinstance(parsed_arguments, (dict, list)):
                st.json(parsed_arguments, expanded=True)
            else:
                st.code(display_scalar(arguments), language="json")
        with st.expander("Raw tool call JSON", expanded=True):
            st.json(tool_call, expanded=True)


def render_jsonl_interactions(row: dict[str, Any]) -> None:
    raw_interactions = row.get("interaction_str") or row.get("interactions_str") or []
    interaction_numbers = row.get("interaction_numbers") or []
    if not raw_interactions or raw_interactions == [""]:
        st.info("No interactions are attached to this row.")
        return
    parsed_interactions = [parse_interaction(item) for item in raw_interactions]
    options = []
    for position, interaction in enumerate(parsed_interactions):
        number = (
            interaction_numbers[position]
            if position < len(interaction_numbers)
            else position
        )
        options.append((position, number, interaction_label(number, interaction)))
    labels = [option[2] for option in options]
    row_key = f"{row.get('_line_number', '')}_{row.get('instance_id', '')}_{row.get('file_name', '')}"
    offset_key = f"interaction_offset_{row_key}"
    label_key = f"selected_interaction_label_{row_key}"
    selected_offset = render_navigation_buttons(offset_key, label_key, labels, f"interaction_nav_{row_key}")
    selected_label = st.selectbox(
        "Interaction", labels, index=selected_offset,
        key=label_key, label_visibility="collapsed",
    )
    selected_position = next(option[0] for option in options if option[2] == selected_label)
    st.session_state[offset_key] = selected_position
    selected = parsed_interactions[selected_position]
    role_cols = st.columns(3)
    role_cols[0].metric("Number", options[selected_position][1])
    role_cols[1].metric("Role", display_scalar(selected.get("role", "")))
    role_cols[2].metric("Tool call id", display_scalar(selected.get("tool_call_id", "")))
    content = display_scalar(selected.get("content", ""))
    if content:
        st.subheader("Content")
        render_wrapped_content(content)
    tool_calls = selected.get("tool_calls")
    if tool_calls:
        render_tool_calls(tool_calls)
    with st.expander("Raw interaction JSON"):
        st.json(selected, expanded=True)


def run_jsonl_mode() -> None:
    """Existing JSONL explorer."""
    jsonl_files = discover_jsonl_files(DATA_DIR)
    compatible_files = [path for path in jsonl_files if is_compatible_jsonl(path)]

    with st.sidebar:
        st.header("Data")
        st.caption(f"Data directory: `{DATA_DIR}`")
        if jsonl_files and not compatible_files:
            st.warning("No compatible JSONL files were found.")
        if not jsonl_files:
            st.error("No `.jsonl` files were found in the data directory.")
            return
        if not compatible_files:
            return
        selected_path = st.selectbox(
            "JSONL file", compatible_files, format_func=lambda p: p.name
        )
        if st.button("Clear load cache"):
            load_jsonl.clear()
            st.rerun()

    if not compatible_files:
        return

    rows, errors = load_jsonl(str(selected_path), selected_path.stat().st_mtime_ns)
    if errors:
        with st.expander(f"Skipped {len(errors)} invalid line(s)", expanded=False):
            st.write("\n".join(errors))
    if not rows:
        st.warning("The selected file did not contain any valid records.")
        return
    missing_fields = sorted(REQUIRED_FIELDS_JSONL - set(rows[0]))
    if missing_fields:
        st.error(f"Selected file is missing required fields: {', '.join(missing_fields)}")
        return

    summary = summarize_records(rows)
    render_jsonl_metrics(summary, rows)
    filtered = apply_jsonl_filters(summary, rows)

    st.subheader("Records")
    st.caption(f"Showing {len(filtered):,} of {len(summary):,} records")
    display_frame = dataframe_for_display(filtered)
    st.dataframe(display_frame, width="stretch", hide_index=True)

    indexes = filtered["_row_index"].tolist()
    if filtered.empty:
        st.info("No records match the current filters.")
        return

    st.subheader("Selected Record")
    labels = [
        f"{row.line} | {row.instance_id} | {row.file_name} | churn {row.churn}"
        for row in filtered.itertuples(index=False)
    ]
    selected_offset = render_navigation_buttons("record_offset", "selected_record_label", labels, "record_nav")
    selected_label = st.selectbox(
        "Record", labels, index=selected_offset,
        key="selected_record_label", label_visibility="collapsed",
    )
    selected_offset = labels.index(selected_label)
    st.session_state["record_offset"] = selected_offset
    selected_index = indexes[selected_offset]
    selected_row = rows[selected_index]

    metadata_tab, diff_tab, interactions_tab, raw_tab = st.tabs(
        ["Metadata", "Diff", "Interactions", "Raw JSON"]
    )
    with metadata_tab:
        render_jsonl_metadata(selected_row)
    with diff_tab:
        st.code(display_scalar(selected_row.get("file_str", "")), language="diff")
    with interactions_tab:
        render_jsonl_interactions(selected_row)
    with raw_tab:
        raw_record = dict(selected_row)
        raw_record.pop("_line_number", None)
        st.json(raw_record, expanded=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE B — JSON trajectory viewer (uploaded agent run files)
# ═══════════════════════════════════════════════════════════════════════════════

ROLE_COLORS: dict[str, str] = {
    "system":    "#6366f1",   # indigo
    "user":      "#10b981",   # emerald
    "assistant": "#f59e0b",   # amber
    "tool":      "#64748b",   # slate
}

ROLE_BADGE_STYLE = (
    "display:inline-block;"
    "padding:2px 10px;"
    "border-radius:999px;"
    "font-size:0.75rem;"
    "font-weight:600;"
    "letter-spacing:0.05em;"
    "text-transform:uppercase;"
    "color:#fff;"
    "background:{color};"
)


def role_badge(role: str) -> str:
    color = ROLE_COLORS.get(role, "#94a3b8")
    style = ROLE_BADGE_STYLE.format(color=color)
    return f'<span style="{style}">{escape(role)}</span>'


def parse_uploaded_json(raw_bytes: bytes) -> dict[str, Any] | None:
    try:
        return json.loads(raw_bytes)
    except json.JSONDecodeError:
        return None


def is_agent_run_json(data: Any) -> bool:
    """Return True if the dict looks like an OpenHands / agent-run trajectory."""
    return isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list)


def extract_text_from_content(content: Any) -> str:
    """Flatten content field (str or list-of-blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "image_url":
                    url = block.get("image_url", {})
                    if isinstance(url, dict):
                        parts.append(f"[image: {url.get('url', '')}]")
                    else:
                        parts.append(f"[image: {url}]")
                elif btype == "tool_result":
                    inner = block.get("content", [])
                    parts.append(extract_text_from_content(inner))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return display_scalar(content)


def message_summary(msg: dict[str, Any], index: int) -> str:
    role = msg.get("role", "?")
    text = extract_text_from_content(msg.get("content", "")).replace("\n", " ")
    tool_calls = msg.get("tool_calls") or []
    if tool_calls and isinstance(tool_calls, list):
        fn = tool_calls[0].get("function", {}).get("name", "")
        snippet = f"[tool: {fn}]" if fn else "[tool call]"
    else:
        snippet = text[:80] + ("..." if len(text) > 80 else "")
    return f"{index} | {role} | {snippet}"


def render_json_run_metrics(data: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    from collections import Counter
    roles = Counter(m.get("role", "?") for m in messages)
    response = data.get("response", {})
    usage = response.get("usage", {})

    cols = st.columns(6)
    cols[0].metric("Messages", len(messages))
    cols[1].metric("Model", response.get("model", "—"))
    cols[2].metric("Cost", f"${data.get('cost', 0):.4f}" if data.get("cost") is not None else "—")
    cols[3].metric("Total tokens", f"{usage.get('total_tokens', 0):,}" if usage else "—")
    cols[4].metric("Cached tokens", f"{usage.get('cache_read_input_tokens', 0):,}" if usage else "—")

    ts = data.get("timestamp")
    if ts:
        dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
        cols[5].metric("Timestamp", dt.strftime("%Y-%m-%d %H:%M UTC"))
    else:
        cols[5].metric("Timestamp", "—")


def render_json_message(msg: dict[str, Any], index: int) -> None:
    role = msg.get("role", "unknown")
    content_raw = msg.get("content", "")
    tool_calls = msg.get("tool_calls") or []
    tool_call_id = msg.get("tool_call_id", "")
    fn_name = msg.get("name", "")

    # Header row
    header_parts = [role_badge(role)]
    if fn_name:
        header_parts.append(f'<span style="color:#94a3b8;font-size:0.8rem;">fn: <code>{escape(fn_name)}</code></span>')
    if tool_call_id:
        header_parts.append(f'<span style="color:#94a3b8;font-size:0.8rem;">id: <code>{escape(tool_call_id)}</code></span>')
    st.html(" &nbsp;".join(header_parts))

    # Content
    if isinstance(content_raw, list):
        for block in content_raw:
            if not isinstance(block, dict):
                render_wrapped_content(str(block))
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    render_wrapped_content(text)
            elif btype == "image_url":
                url = block.get("image_url", {})
                if isinstance(url, dict):
                    url_str = url.get("url", "")
                else:
                    url_str = str(url)
                if url_str.startswith("data:"):
                    st.image(url_str, caption="[embedded image]")
                else:
                    st.markdown(f"🖼️ [image]({url_str})")
            elif btype == "tool_result":
                inner_text = extract_text_from_content(block.get("content", []))
                if inner_text:
                    render_wrapped_content(inner_text)
            else:
                with st.expander(f"Block: {btype}", expanded=False):
                    st.json(block)
    elif isinstance(content_raw, str) and content_raw.strip():
        render_wrapped_content(content_raw)

    # Tool calls (assistant messages that invoke tools)
    if tool_calls:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")
            args_raw = fn.get("arguments", "")
            args_parsed = try_parse_json(args_raw)

            with st.expander(f"🔧 Tool call: `{tc_name}`", expanded=True):
                if isinstance(args_parsed, dict):
                    st.json(args_parsed, expanded=True)
                else:
                    st.code(display_scalar(args_raw), language="json")


def run_json_upload_mode(uploaded_files: list[Any]) -> None:
    """Viewer for one or more uploaded agent-run JSON files."""

    # Parse all files
    parsed: list[tuple[str, dict[str, Any]]] = []
    for uf in uploaded_files:
        data = parse_uploaded_json(uf.read())
        if data is None:
            st.sidebar.error(f"⚠️ {uf.name}: not valid JSON")
            continue
        if not is_agent_run_json(data):
            st.sidebar.warning(f"⚠️ {uf.name}: missing `messages` array")
            continue
        parsed.append((uf.name, data))

    if not parsed:
        st.error("None of the uploaded files could be parsed as agent-run trajectories.")
        return

    # File selector in sidebar
    with st.sidebar:
        st.header("Uploaded files")
        file_names = [name for name, _ in parsed]
        selected_name = st.selectbox("Select file", file_names)

    data = next(d for name, d in parsed if name == selected_name)
    messages: list[dict[str, Any]] = data.get("messages", [])

    st.subheader(f"📄 {selected_name}")
    render_json_run_metrics(data, messages)

    st.divider()

    # Message filter in sidebar
    with st.sidebar:
        st.header("Filters")
        from collections import Counter
        all_roles = sorted({m.get("role", "?") for m in messages})
        selected_roles = st.multiselect("Roles", all_roles, default=all_roles)
        search_query = st.text_input("Search content", placeholder="grep messages...")

        tool_names_all = sorted({
            tc.get("function", {}).get("name", "")
            for m in messages
            for tc in (m.get("tool_calls") or [])
            if isinstance(tc, dict) and tc.get("function", {}).get("name")
        })
        selected_tools = st.multiselect("Tool calls", tool_names_all)

    # Apply filters
    filtered_messages = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        if role not in selected_roles:
            continue
        if selected_tools:
            tc_names = {
                tc.get("function", {}).get("name", "")
                for tc in (msg.get("tool_calls") or [])
                if isinstance(tc, dict)
            }
            if not tc_names.intersection(selected_tools):
                continue
        if search_query:
            full_text = extract_text_from_content(msg.get("content", "")) + display_scalar(msg.get("tool_calls", ""))
            if search_query.lower() not in full_text.lower():
                continue
        filtered_messages.append((i, msg))

    st.caption(f"Showing {len(filtered_messages)} of {len(messages)} messages")

    if not filtered_messages:
        st.info("No messages match the current filters.")
        return

    # Navigation
    labels = [message_summary(msg, orig_i) for orig_i, msg in filtered_messages]
    selected_offset = render_navigation_buttons(
        "json_msg_offset", "json_msg_label", labels, "json_msg_nav"
    )
    selected_label = st.selectbox(
        "Message", labels, index=selected_offset,
        key="json_msg_label", label_visibility="collapsed",
    )
    selected_offset = labels.index(selected_label)
    st.session_state["json_msg_offset"] = selected_offset

    orig_index, selected_msg = filtered_messages[selected_offset]

    # Render the selected message
    st.markdown(f"**Message {orig_index + 1} of {len(messages)}**")
    msg_tab, raw_tab = st.tabs(["Message", "Raw JSON"])
    with msg_tab:
        render_json_message(selected_msg, orig_index)
    with raw_tab:
        st.json(selected_msg, expanded=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("Trajectory Explorer")

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Source",
            ["Upload JSON files", "JSONL from data/"],
            help="Upload individual agent-run JSON files, or browse JSONL files already in the data/ directory.",
        )

        if mode == "Upload JSON files":
            uploaded_files = st.file_uploader(
                "Drop trajectory JSON files here",
                type=["json"],
                accept_multiple_files=True,
                help="Accepts OpenHands-style agent run JSON files with a top-level `messages` array.",
            )

    if mode == "Upload JSON files":
        if not uploaded_files:
            st.info(
                "👆 Upload one or more agent-run **JSON** files using the sidebar uploader.\n\n"
                "Expected format: `{ messages: [...], response: {...}, cost: ..., timestamp: ... }`"
            )
            return
        run_json_upload_mode(uploaded_files)
    else:
        run_jsonl_mode()


if __name__ == "__main__":
    main()
