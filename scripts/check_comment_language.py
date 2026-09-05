"""模块职责：扫描 Python 源码中的自然语言注释与 docstring，报告未使用中文的违规项。

不负责：修改任何文件；检查注释的内容质量（准确性与信息密度由人工评审承担）。
关键约束：只依赖标准库（tokenize/ast）；docstring 节点覆盖 module/class/function/method。
依赖关系：供注释规范化计划（docs/project/CODE_COMMENT_STANDARDIZATION_PLAN_20260905.md）
          的各工作包与最终验收使用；可用命令行参数指定文件，缺省扫描根目录、
          scripts/ 与 presentation/ 下的全部 Python 文件。

违规判定：
- 注释（tokenize 的 COMMENT token）：不含中文字符且含 ASCII 字母，且未命中豁免规则。
- docstring（ast 提取的模块/类/函数文档字符串）：不含中文字符。
豁免规则（计划 1.2 节）：shebang、编码声明、lint/type 等 pragma、含 URL 的引用信息。
shape 记号与库名等英文技术token必须嵌入中文语境，单独构成自然语言注释仍判违规。

用法：
    python scripts/check_comment_language.py                # 扫描缺省文件集
    python scripts/check_comment_language.py a.py b.py      # 扫描指定文件

退出码：发现违规返回 1，否则 0。
"""

import ast
import io
import os
import re
import sys
import tokenize

# 中文字符判定范围：CJK 统一表意文字基本区 + 扩展 A 区。
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
# 含 URL 视为论文/参考实现引用信息，按计划 1.2 节豁免。
_URL = re.compile(r"https?://")
# pragma 前缀：去掉行首 "# " 后按小写前缀匹配。
_PRAGMA_PREFIXES = (
    "!",
    "-*- coding",
    "coding:",
    "coding =",
    "noqa",
    "type:",
    "pylint:",
    "pragma:",
    "fmt:",
    "yapf:",
    "isort:",
    "ruff:",
    "mypy:",
    "nosec",
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_files():
    """返回缺省扫描集：根目录、scripts/、presentation/ 下的全部 .py 文件。"""
    files = []
    for rel_dir in (".", "scripts", "presentation"):
        abs_dir = os.path.join(_ROOT, rel_dir)
        for name in sorted(os.listdir(abs_dir)):
            if name.endswith(".py"):
                files.append(os.path.normpath(os.path.join(abs_dir, name)))
    return files


def _is_commented_out_code(body):
    """判断注释内容是否为被注释掉的可执行代码（非自然语言，不受语言规范约束）。

    判定规则：整段内容能被 ast.parse 解析，且每个语句都不是"裸名称/裸字面量"。
    复合语句头（如 `# if cond:`、`# with ctx():`）单行无法成句，先补 `pass` 再试；
    这样被禁用代码块的续行也能豁免。这样 `# helpers`、`# sampling` 这类裸英文短语仍判违规，
    而 `# x = f(...)`、`# assert ...`、`# with ...:` 等真实语句被豁免。被禁用但有意保留的
    历史代码块（AGENTS.md 明确保护，如 diffusion.py 的 self-conditioning 块）因此不计入
    自然语言注释违规；是否保留该代码块由 AGENTS.md 与代码评审决定，不归本扫描器。
    """
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError):
        # 复合语句头（if/with/for 等缺函数体）补一个 pass 再尝试。
        try:
            tree = ast.parse(body.strip() + "\n    pass")
        except (SyntaxError, ValueError):
            return False
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(
            stmt.value, (ast.Name, ast.Constant)
        ):
            return False
    return True


def _comment_exempt(text):
    """判断一条注释是否命中豁免规则（工具指令、引用信息或注释代码，而非自然语言说明）。"""
    body = text.lstrip("#").strip()
    low = body.lower()
    for prefix in _PRAGMA_PREFIXES:
        if low.startswith(prefix):
            return True
    if _URL.search(body):
        return True
    if _is_commented_out_code(body):
        return True
    return False


def _scan_file(path):
    """返回 (注释违规列表[(行号, 文本)], docstring 违规列表[(行号, 文本)], 统计 dict)。"""
    with open(path, "rb") as handle:
        source = handle.read()

    comment_total = 0
    comment_bad = []
    try:
        for tok in tokenize.tokenize(io.BytesIO(source).readline):
            if tok.type == tokenize.COMMENT:
                comment_total += 1
                if _CJK.search(tok.string):
                    continue
                if not re.search(r"[A-Za-z]", tok.string):
                    # 纯符号/数字注释（如 "#####"）不属于自然语言。
                    continue
                if _comment_exempt(tok.string):
                    continue
                comment_bad.append((tok.start[0], tok.string))
    except tokenize.TokenError as exc:  # pragma: no cover - 语法错误文件单独报错
        raise SystemExit(f"{path}: tokenize 失败：{exc}")

    tree = ast.parse(source, filename=path)
    doc_total = 0
    doc_bad = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        doc_total += 1
        if not _CJK.search(doc):
            first = doc.strip().splitlines()[0] if doc.strip() else ""
            # 模块节点没有 lineno；docstring 存在时 body 首节点即其表达式。
            lineno = node.body[0].lineno
            doc_bad.append((lineno, first))
    return comment_bad, doc_bad, {
        "comments": comment_total,
        "docstrings": doc_total,
    }


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    files = argv[1:] or [
        f for f in _default_files()
        if os.path.normcase(f) != os.path.normcase(os.path.abspath(__file__))
    ]
    files = [os.path.abspath(f) for f in files]

    total_comment_bad = 0
    total_doc_bad = 0
    for path in files:
        comment_bad, doc_bad, stats = _scan_file(path)
        total_comment_bad += len(comment_bad)
        total_doc_bad += len(doc_bad)
        print(
            f"{os.path.relpath(path, _ROOT)}: "
            f"comment_tokens={stats['comments']} "
            f"english_comments={len(comment_bad)} "
            f"docstrings={stats['docstrings']} "
            f"english_docstrings={len(doc_bad)}"
        )
        for lineno, text in comment_bad:
            snippet = text if len(text) <= 90 else text[:87] + "..."
            print(f"    L{lineno}: {snippet}")
        for lineno, text in doc_bad:
            snippet = text if len(text) <= 90 else text[:87] + "..."
            print(f"    L{lineno} (docstring): {snippet}")

    print(
        f"TOTAL: english_comments={total_comment_bad} "
        f"english_docstrings={total_doc_bad}"
    )
    return 1 if (total_comment_bad or total_doc_bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
