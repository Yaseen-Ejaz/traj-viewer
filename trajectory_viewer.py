from __future__ import annotations

import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Trajectory Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_KEYWORDS: list[str] = [
    "image",
    "screenshot",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "bitmap",
    "visual",
    "picture",
    "photo",
    "figure",
    "diagram",
    "graph",
    "chart",
    "render",
    "thumbnail",
    "pixel",
    "icon",
    "analyze.*image",
    "look.*at.*image",
    "view.*image",
    "see.*image",
    "examining.*image",
    "reading.*image",
]

ROLE_COLORS: dict[str, str] = {
    "system":    "#6c757d",
    "user":      "#0d6efd",
    "assistant": "#6f42c1",
    "tool":      "#fd7e14",
}


def role_badge(role: str) -> str:
    color = ROLE_COLORS.get(role, "#888")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;'
        f'letter-spacing:.03em">{role.upper()}</span>'
    )


def keyword_matches(text: str, keywords: list[str]) -> list[str]:
    hits: list[str] = []
    t = text.lower()
    for kw in keywords:
        try:
            if re.search(kw, t, re.IGNORECASE):
                hits.append(kw)
        except re.error:
            if kw in t:
                hits.append(kw)
    return hits


def highlight_keywords(text: str, keywords: list[str]) -> str:
    """Return HTML with keyword occurrences highlighted."""
    result = escape(text)
    for kw in keywords:
        try:
            pattern = re.compile(f"({kw})", re.IGNORECASE)
            result = pattern.sub(
                r'<mark style="background:#fff3cd;padding:0 2px;border-radius:3px">\1</mark>',
                result,
            )
        except re.error:
            escaped_kw = escape(kw)
            result = result.replace(
                escaped_kw,
                f'<mark style="background:#fff3cd;padding:0 2px;border-radius:3px">{escaped_kw}</mark>',
            )
    return result


def fmt_timestamp(ts: float | None) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def extract_text_from_content(content: Any) -> str:
    """Pull all plain text out of a content block (list or string)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(block))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def full_message_text(msg: dict) -> str:
    """All searchable text from a message (content + tool calls)."""
    parts: list[str] = [extract_text_from_content(msg.get("content", ""))]
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        parts.append(fn.get("name", ""))
        parts.append(fn.get("arguments", ""))
    return " ".join(parts)


def image_blocks(msg: dict) -> list[dict]:
    """Return all image_url blocks inside a message."""
    content = msg.get("content", [])
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "image_url"]


def has_image(msg: dict) -> bool:
    return len(image_blocks(msg)) > 0



def analyze_trajectory(data: dict, keywords: list[str]) -> dict:
    messages = data.get("messages", [])
    role_counts: dict[str, int] = {}
    img_msg_indices: list[int] = []
    kw_msg_indices: list[int] = []
    total_tool_calls = 0
    base64_imgs = 0
    external_imgs = 0
    msg_analysis: list[dict] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        total_tool_calls += len(msg.get("tool_calls", []))

        imgs = image_blocks(msg)
        msg_has_img = len(imgs) > 0
        if msg_has_img:
            img_msg_indices.append(i)
            for b in imgs:
                url = b.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    base64_imgs += 1
                else:
                    external_imgs += 1

        all_text = full_message_text(msg)
        hits = keyword_matches(all_text, keywords)
        msg_has_kw = len(hits) > 0
        if msg_has_kw:
            kw_msg_indices.append(i)

        msg_analysis.append({
            "index": i,
            "role": role,
            "has_image": msg_has_img,
            "image_blocks": imgs,
            "kw_hits": hits,
            "has_kw": msg_has_kw,
            "tool_calls": msg.get("tool_calls", []),
            "content": msg.get("content", []),
            "tool_call_id": msg.get("tool_call_id"),
            "name": msg.get("name"),
            "all_text": all_text,
            "snippet": all_text.replace("\n", " ").strip()[:120],
        })

    return {
        "total": len(messages),
        "role_counts": role_counts,
        "total_tool_calls": total_tool_calls,
        "img_msg_count": len(img_msg_indices),
        "kw_msg_count": len(kw_msg_indices),
        "img_msg_indices": img_msg_indices,
        "kw_msg_indices": kw_msg_indices,
        "base64_imgs": base64_imgs,
        "external_imgs": external_imgs,
        "messages": msg_analysis,
        "cost": data.get("cost"),
        "timestamp": data.get("timestamp"),
        "model": data.get("response", {}).get("model", ""),
    }



def render_summary_cards(analysis: dict) -> None:
    total         = analysis["total"]
    img_count     = analysis["img_msg_count"]
    kw_count      = analysis["kw_msg_count"]
    tool_calls    = analysis["total_tool_calls"]
    cost          = analysis["cost"]
    b64           = analysis["base64_imgs"]
    ext           = analysis["external_imgs"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Messages", f"{total:,}")
    col2.metric("Tool calls", f"{tool_calls:,}")
    col3.metric("🖼 Msgs w/ images", img_count,
                delta="⚠ check needed" if img_count > 0 else None,
                delta_color="inverse" if img_count > 0 else "normal")
    col4.metric("🔍 Keyword matches", kw_count,
                delta="⚠ check needed" if kw_count > 0 else None,
                delta_color="inverse" if kw_count > 0 else "normal")
    col5.metric("Base64 imgs", b64)
    col6.metric("Cost", f"${cost:.5f}" if cost else "—")

    if analysis.get("model"):
        st.caption(
            f"Model: `{analysis['model']}`  |  "
            f"Timestamp: {fmt_timestamp(analysis['timestamp'])}  |  "
            f"External img URLs: {ext}  |  "
            f"Base64 screenshots: {b64}"
        )


def render_alert_banner(analysis: dict) -> None:
    img_count = analysis["img_msg_count"]
    kw_count  = analysis["kw_msg_count"]

    if img_count == 0 and kw_count == 0:
        st.success("✅ No image content or image-related keywords detected in this trajectory.")
        return

    if img_count > 0:
        st.warning(
            f"⚠️ **{img_count} message(s)** contain actual image blocks "
            f"({analysis['external_imgs']} external URL, {analysis['base64_imgs']} base64 screenshot). "
            f"The agent was sent or received image data."
        )
    if kw_count > 0:
        st.error(
            f"🔍 **{kw_count} message(s)** contain image-related keywords in text content or tool arguments. "
            f"Review highlighted messages below."
        )


def render_image_block(block: dict, idx: int) -> None:
    url = block.get("image_url", {}).get("url", "")
    if url.startswith("data:"):
        st.caption(f"📸 Screenshot (base64) #{idx+1}")
        st.image(url, use_container_width=False, width=600)
    else:
        st.caption(f"🖼 External image URL #{idx+1}")
        st.code(url, language=None)
        try:
            st.image(url, use_container_width=False, width=400)
        except Exception:
            st.warning("Could not load image preview.")


_PRE_STYLE = (
    "font-family:monospace;font-size:.8rem;line-height:1.45;"
    "white-space:pre-wrap;word-break:break-word;overflow-x:auto;"
    "padding:.6rem .75rem;border-radius:.4rem;"
    "border:1px solid rgba(128,128,128,0.2);"
)


def _render_highlighted(text: str, kw_hits: list[str]) -> None:
    """Render text with keyword highlights using st.markdown (inherits theme bg)."""
    highlighted = highlight_keywords(text, kw_hits)
    st.markdown(
        f'<div style="{_PRE_STYLE}">{highlighted}</div>',
        unsafe_allow_html=True,
    )


def render_content_block(block: dict | str, kw_hits: list[str], show_kw: bool) -> None:
    if isinstance(block, str):
        if show_kw and kw_hits:
            _render_highlighted(block, kw_hits)
        else:
            st.code(block, language=None)
        return

    if not isinstance(block, dict):
        st.json(block)
        return

    btype = block.get("type")
    if btype == "text":
        text = block.get("text", "")
        if show_kw and kw_hits:
            _render_highlighted(text, kw_hits)
        else:
            st.code(text, language=None)
    elif btype == "image_url":
        pass  # handled separately above the fold
    else:
        st.json(block)


def render_tool_call(tc: dict, kw_hits: list[str], show_kw: bool) -> None:
    fn   = tc.get("function", {})
    name = fn.get("name", "unknown")
    args = fn.get("arguments", "")
    try:
        parsed = json.loads(args)
        args_display = json.dumps(parsed, indent=2)
    except Exception:
        args_display = args

    with st.expander(f"🔧 Tool call: `{name}`", expanded=False):
        st.caption(f"ID: `{tc.get('id', '—')}`  |  Type: `{tc.get('type', '—')}`")
        if show_kw and kw_hits:
            _render_highlighted(args_display, kw_hits)
        else:
            st.code(args_display, language="json")


def render_message(msg_info: dict, show_kw: bool) -> None:
    role      = msg_info["role"]
    has_img   = msg_info["has_image"]
    kw_hits   = msg_info["kw_hits"]
    has_kw    = msg_info["has_kw"]
    content   = msg_info["content"]
    tool_calls = msg_info["tool_calls"]
    imgs      = msg_info["image_blocks"]
    idx       = msg_info["index"]

    # Build expander label
    flags = ""
    if has_img:
        flags += " 🖼"
    if has_kw:
        flags += " 🔍"
    snippet = msg_info["snippet"][:80] + ("…" if len(msg_info["snippet"]) > 80 else "")
    label = f"#{idx} [{role.upper()}]{flags}  {snippet}"

    # Colour the border via container
    border_color = "#dc3545" if (has_img or has_kw) else "#dee2e6"
    left_border  = "4px" if (has_img or has_kw) else "0px"

    with st.container(border=True):
        # Header row
        header_col, flag_col = st.columns([8, 2])
        with header_col:
            st.markdown(
                f'{role_badge(role)}'
                f'<span style="margin-left:8px;font-size:.8rem;color:#6c757d">#{idx}</span>',
                unsafe_allow_html=True,
            )
        with flag_col:
            flags_html = ""
            if has_img:
                flags_html += '<span style="color:#fd7e14;font-weight:600">🖼 IMAGE</span> '
            if has_kw:
                flags_html += '<span style="color:#dc3545;font-weight:600">🔍 KEYWORD</span>'
            if flags_html:
                st.markdown(flags_html, unsafe_allow_html=True)

        # Keyword hits summary
        if has_kw and show_kw:
            st.caption(f"Keyword hits: {', '.join(f'`{h}`' for h in kw_hits)}")

        # Image blocks first (most important)
        for i, img in enumerate(imgs):
            render_image_block(img, i)

        # Content blocks
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    continue  # already shown above
                render_content_block(block, kw_hits, show_kw)
        elif content:
            render_content_block(content, kw_hits, show_kw)

        # Tool calls
        for tc in tool_calls:
            render_tool_call(tc, kw_hits, show_kw)

        # Tool result metadata
        if msg_info.get("tool_call_id"):
            st.caption(
                f"Tool result for: `{msg_info['tool_call_id']}`"
                + (f"  |  name: `{msg_info['name']}`" if msg_info.get("name") else "")
            )



def build_sidebar() -> tuple[list[str], bool, bool, str, str, bool]:
    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("Keywords to detect")
        raw_kw = st.text_area(
            "One per line",
            value="\n".join(DEFAULT_KEYWORDS),
            height=220,
            help="Supports Python regex. Each line is one pattern.",
        )
        keywords = [k.strip() for k in raw_kw.splitlines() if k.strip()]
        show_kw_highlights = st.checkbox("Highlight keywords in text", value=True)

        st.divider()
        st.subheader("Filters")
        role_filter = st.selectbox(
            "Role",
            ["all", "system", "user", "assistant", "tool"],
        )
        content_filter = st.selectbox(
            "Show",
            ["all messages", "images only", "keyword hits only", "images or keyword hits"],
        )
        search_query = st.text_input("Search text", placeholder="grep the content...")

        st.divider()
        expand_all = st.checkbox("Expand all messages", value=False)

    return keywords, show_kw_highlights, expand_all, role_filter, content_filter, search_query



def main() -> None:
    st.title("🔍 Trajectory Viewer")
    st.caption("Upload OpenHands/SWE-bench trajectory JSON files to inspect messages and detect image content.")

    keywords, show_kw, expand_all, role_filter, content_filter, search_query = build_sidebar()

    uploaded = st.file_uploader(
        "Upload trajectory JSON file(s)",
        type=["json"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("Upload one or more trajectory `.json` files to get started.")
        return

    # File selector when multiple uploaded
    if len(uploaded) > 1:
        file_names = [f.name for f in uploaded]
        selected_name = st.selectbox("Select file", file_names)
        active_file = next(f for f in uploaded if f.name == selected_name)
    else:
        active_file = uploaded[0]
        st.caption(f"Loaded: `{active_file.name}`")

    # Parse
    try:
        raw = json.loads(active_file.read())
    except Exception as exc:
        st.error(f"Failed to parse JSON: {exc}")
        return

    if "messages" not in raw:
        st.error("This file does not contain a `messages` array — is it the right format?")
        return

    analysis = analyze_trajectory(raw, keywords)

    # Overview
    st.divider()
    render_summary_cards(analysis)
    st.divider()
    render_alert_banner(analysis)
    st.divider()

    # Role distribution
    with st.expander("Role distribution", expanded=False):
        cols = st.columns(len(analysis["role_counts"]))
        for col, (role, count) in zip(cols, analysis["role_counts"].items()):
            col.metric(role, count)

    # Message list
    st.subheader(f"Messages ({analysis['total']})")

    msgs = analysis["messages"]

    # Apply filters
    if role_filter != "all":
        msgs = [m for m in msgs if m["role"] == role_filter]

    if content_filter == "images only":
        msgs = [m for m in msgs if m["has_image"]]
    elif content_filter == "keyword hits only":
        msgs = [m for m in msgs if m["has_kw"]]
    elif content_filter == "images or keyword hits":
        msgs = [m for m in msgs if m["has_image"] or m["has_kw"]]

    if search_query:
        q = search_query.lower()
        msgs = [m for m in msgs if q in m["all_text"].lower()]

    st.caption(f"Showing {len(msgs)} of {analysis['total']} messages")

    if not msgs:
        st.info("No messages match the current filters.")
        return

    for msg_info in msgs:
        render_message(msg_info, show_kw)


if __name__ == "__main__":
    main()