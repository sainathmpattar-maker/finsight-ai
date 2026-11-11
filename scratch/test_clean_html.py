import textwrap

def clean_html_str(html: str) -> str:
    if not html:
        return ""
    dedented = textwrap.dedent(html).strip()
    cleaned_lines = [line.strip() for line in dedented.splitlines() if line.strip()]
    result = []
    for i, line in enumerate(cleaned_lines):
        result.append(line)
        if i < len(cleaned_lines) - 1:
            next_line = cleaned_lines[i + 1]
            if not line.endswith(">") and not next_line.startswith("<"):
                result.append(" ")
    return "".join(result)

sample = """
    <div class="genz-glass-card">
        <p>
            Hello
            world
        </p>
        <span>Tag</span>
    </div>
"""
print(clean_html_str(sample))

