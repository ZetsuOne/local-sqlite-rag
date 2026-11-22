from __future__ import annotations
import re
import html
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
def extract_text(file_path: Path) -> tuple[str, str]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    name = path.name.lower()
    if name.endswith(".fb2.zip") or (suffix == ".zip" and ".fb2" in name):
        return _parse_fb2_zip(path)
    if suffix == ".fb2":
        return _parse_fb2(path.read_bytes(), path.stem)
    if suffix == ".epub":
        return _parse_epub(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return path.stem, _clean(text)
    raise ValueError(f"Unsupported format: {path}")
def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag
def _parse_fb2_zip(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as zf:
        fb2_name = next(
            (n for n in zf.namelist() if n.lower().endswith(".fb2")),
            None,
        )
        if fb2_name is None:
            raise ValueError(f"No .fb2 inside archive: {path}")
        return _parse_fb2(zf.read(fb2_name), path.stem.replace(".fb2", ""))
def _parse_fb2_fallback(data: bytes, fallback_title: str) -> tuple[str, str]:
    head = data[:512].decode("ascii", errors="ignore")
    m = re.search(r'encoding=["\']([^"\']+)["\']', head, re.IGNORECASE)
    enc = m.group(1) if m else "utf-8"
    try:
        raw = data.decode(enc, errors="replace")
    except LookupError:
        raw = data.decode("utf-8", errors="replace")

    title_m = re.search(r"<book-title[^>]*>(.*?)</book-title>", raw, re.IGNORECASE | re.DOTALL)
    title = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip() if title_m else fallback_title

    raw = re.sub(r"<binary\b[^>]*>.*?</binary>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    body_m = re.search(r"<body\b[^>]*>.*</body>", raw, re.IGNORECASE | re.DOTALL)
    content = body_m.group(0) if body_m else raw

    content = re.sub(r"</(?:p|v|subtitle|text-author|title)>|<empty-line\s*/?>", "\n\n", content, flags=re.IGNORECASE)
    content = re.sub(r"<[^>]+>", "", content)
    return title or fallback_title, _clean(html.unescape(content))
def _parse_fb2(data: bytes, fallback_title: str) -> tuple[str, str]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return _parse_fb2_fallback(data, fallback_title)

    title = fallback_title
    for el in root.iter():
        if _local(el.tag) == "book-title" and (el.text or "").strip():
            title = el.text.strip()
            break
    bodies = [el for el in root.iter() if _local(el.tag) == "body"]
    roots = bodies or [root]
    parts: list[str] = []
    for body in roots:
        for el in body.iter():
            tag = _local(el.tag)
            if tag == "binary":
                continue
            if tag in {"p", "v", "subtitle", "text-author"}:
                chunk = _fb2_element_text(el).strip()
                if chunk:
                    parts.append(chunk)
            elif tag == "empty-line":
                parts.append("")
    return title, _clean("\n\n".join(parts))
def _fb2_element_text(el: ET.Element) -> str:
    if _local(el.tag) == "binary":
        return ""
    parts: list[str] = []
    if el.text:
        parts.append(el.text)
    for child in el:
        if _local(child.tag) == "binary":
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_fb2_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)
def _parse_epub(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        rootfile = None
        for el in container.iter():
            if _local(el.tag) == "rootfile":
                rootfile = el
                break
        if rootfile is None:
            raise ValueError(f"Invalid EPUB container: {path}")
        opf_path = rootfile.get("full-path")
        if not opf_path:
            raise ValueError(f"Missing OPF path in: {path}")
        opf_dir = str(Path(opf_path).parent).replace("\\", "/")
        if opf_dir == ".":
            opf_dir = ""
        opf = ET.fromstring(zf.read(opf_path))
        title = path.stem
        for el in opf.iter():
            if _local(el.tag) == "title" and (el.text or "").strip():
                title = el.text.strip()
                break
        manifest: dict[str, str] = {}
        for item in opf.iter():
            if _local(item.tag) != "item":
                continue
            item_id, href = item.get("id"), item.get("href")
            if item_id and href:
                full = f"{opf_dir}/{href}" if opf_dir else href
                manifest[item_id] = full.lstrip("/")
        spine_hrefs: list[str] = []
        for itemref in opf.iter():
            if _local(itemref.tag) != "itemref":
                continue
            idref = itemref.get("idref")
            if idref and idref in manifest:
                spine_hrefs.append(manifest[idref])
        parts: list[str] = []
        for href in spine_hrefs:
            try:
                raw = zf.read(href)
            except KeyError:
                continue
            try:
                doc = ET.fromstring(raw)
            except ET.ParseError:
                continue
            texts: list[str] = []
            for el in doc.iter():
                if _local(el.tag) in {"script", "style"}:
                    continue
                if el.text and el.text.strip():
                    texts.append(el.text.strip())
                if el.tail and el.tail.strip():
                    texts.append(el.tail.strip())
            if texts:
                parts.append("\n".join(texts))
        return title, _clean("\n\n".join(parts))
def _parse_pdf(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    title = path.stem
    meta = reader.metadata
    if meta is not None:
        t = getattr(meta, "title", None)
        if not t and hasattr(meta, "get"):
            t = meta.get("/Title")
        if t and str(t).strip():
            title = str(t).strip()
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return title, _clean("\n\n".join(pages))