#!/usr/bin/env python3
"""Documentation coverage analyzer for Python, C++, and CUDA.

Handles:
- Python: AST-based docstring detection
- C++/CUDA: Doxygen-style documentation (/** */, ///, @brief, etc.)

Usage:
    python tools/docstring_audit.py
"""

import os
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Item:
    """Represents a documented code item."""
    name: str
    line: int
    has_doc: bool


@dataclass
class FileStats:
    """Statistics for a single file."""
    path: str
    has_header: bool
    items: List[Item]
    
    @property
    def missing_count(self) -> int:
        return sum(1 for i in self.items if not i.has_doc)


def analyze_python_file(filepath: str) -> Optional[FileStats]:
    """Analyze a Python file for docstring coverage."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception:
        return None
    
    has_header = ast.get_docstring(tree) is not None
    items: List[Item] = []
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            items.append(Item(node.name, node.lineno, doc is not None))
            
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith('_'):
                        doc = ast.get_docstring(item)
                        items.append(Item(f"{node.name}.{item.name}", item.lineno, doc is not None))
                        
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith('_'):
                doc = ast.get_docstring(node)
                items.append(Item(node.name, node.lineno, doc is not None))
    
    return FileStats(filepath, has_header, items)


def has_doxygen_doc(content: str, end_pos: int) -> bool:
    """Check if there's Doxygen documentation before the given position."""
    # Get the content before this position (look back up to 500 chars)
    start_pos = max(0, end_pos - 500)
    before = content[start_pos:end_pos]
    
    # Check for Doxygen-style documentation
    # /** ... */ block
    if re.search(r'/\*\*|\/\*!', before):
        return True
    # /// single-line comment (not part of other things)
    if re.search(r'\n\s*///[^/]', before):  # /// that isn't ////
        return True
    # \<!\b  (some use <! instead)
    if re.search(r'<\s*!\s*-{2}', before):
        return True
    
    return False


def analyze_cpp_file(filepath: str) -> Optional[FileStats]:
    """Analyze a C++/CUDA file for Doxygen-style documentation."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return None
    
    items: List[Item] = []
    
    # Check for header comment (file-level Doxygen)
    has_header = bool(re.search(r'/\*\*[\s\S]{50,}\*/', content[:2000])) or \
                 bool(re.search(r'///\s*@file', content[:500], re.MULTILINE))
    
    # Find function declarations using multiple patterns
    patterns = [
        # Standard function: return_type name(args);
        r'(?:^|\n)(?:\s*)(?:\w[\w\s\*&]*?\s+)([a-zA-Z_]\w*)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*(?:noexcept)?\s*(?:;\s*$|{\s*$|=?\s*0\s*;)',
        # CUDA kernels: __global__ void name(...)
        r'(?:__global__|__device__|__host__)\s+(?:inline\s+)?(?:\w+)\s+(\w+)\s*\([^)]*\)',
        # Template functions
        r'template\s*<[^>]+>\s*(?:\w+)\s+(\w+)\s*\([^)]*\)',
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Skip comments and preprocessor
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        if stripped.startswith('#'):
            continue
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, line, re.MULTILINE))
            for match in matches:
                name = match.group(1) if match.lastindex else None
                if name:
                    # Filter out keywords and special names
                    skip_keywords = {
                        'if', 'else', 'while', 'for', 'switch', 'case', 'do', 'return',
                        'sizeof', 'typedef', 'class', 'struct', 'enum', 'union',
                        'namespace', 'using', 'static', 'const', 'inline', 'virtual',
                        'explicit', 'override', 'final', 'delete', 'default',
                        'nullptr', 'true', 'false', 'operator', 'and', 'or', 'not',
                        'public', 'private', 'protected', 'friend', 'register', 'volatile'
                    }
                    if name in skip_keywords:
                        continue
                    if name.startswith('_') or name.startswith('~'):
                        continue
                    # Skip if looks like a type (all caps or CamelCase for types)
                    if name[0].isupper() and '_' not in name:
                        continue  # Likely a type declaration
                    
                    # Check for documentation before this line
                    pos = content.find(line)
                    has_doc = has_doxygen_doc(content, pos)
                    items.append(Item(name, i + 1, has_doc))
                    break  # Only match first pattern per line
    
    return FileStats(filepath, has_header, items)


def main():
    """Run the documentation audit."""
    print("=" * 80)
    print("DOCUMENTATION COVERAGE AUDIT")
    print("=" * 80)
    
    # Python audit
    print("\n" + "=" * 80)
    print("PYTHON DOCSTRING COVERAGE - swe2d package")
    print("=" * 80)
    
    swe2d_dir = 'swe2d'
    py_stats = {
        'files': 0,
        'missing_module': 0,
        'missing_class': 0,
        'missing_func': 0,
        'total_class': 0,
        'total_func': 0,
        'file_details': []
    }
    
    for root, dirs, files in os.walk(swe2d_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                result = analyze_python_file(filepath)
                if result:
                    py_stats['files'] += 1
                    if not result.has_header:
                        py_stats['missing_module'] += 1
                    
                    for item in result.items:
                        if '.' in item.name:  # method
                            py_stats['total_func'] += 1
                            if not item.has_doc:
                                py_stats['missing_func'] += 1
                        else:  # class or function
                            if re.match(r'^[A-Z]', item.name):  # Class
                                py_stats['total_class'] += 1
                                if not item.has_doc:
                                    py_stats['missing_class'] += 1
                            else:  # Function
                                py_stats['total_func'] += 1
                                if not item.has_doc:
                                    py_stats['missing_func'] += 1
                    
                    if result.missing_count > 0:
                        py_stats['file_details'].append(result)
    
    py_total = py_stats['total_class'] + py_stats['total_func']
    py_missing = py_stats['missing_class'] + py_stats['missing_func']
    py_coverage = ((py_total - py_missing) / py_total * 100) if py_total > 0 else 0
    
    print(f"\nFiles analyzed: {py_stats['files']}")
    print(f"Module docstrings missing: {py_stats['missing_module']}/{py_stats['files']}")
    print(f"Class docstrings missing: {py_stats['missing_class']}/{py_stats['total_class']}")
    print(f"Function docstrings missing: {py_stats['missing_func']}/{py_stats['total_func']}")
    print(f"\nClass + Function coverage: {py_coverage:.1f}%")
    
    # Show files needing attention
    print("\n" + "-" * 80)
    print("PYTHON FILES NEEDING ATTENTION (sorted by missing count)")
    print("-" * 80)
    py_stats['file_details'].sort(key=lambda x: x.missing_count, reverse=True)
    for item in py_stats['file_details'][:20]:
        total = len(item.items)
        missing = item.missing_count
        cov = ((total - missing) / total * 100) if total > 0 else 0
        print(f"\n  {item.path}")
        print(f"    {missing}/{total} items missing ({cov:.0f}% coverage)")
        for i in item.items:
            if not i.has_doc:
                print(f"      - {i.name} (line {i.line})")
    
    # C++ audit
    print("\n" + "=" * 80)
    print("C++/CUDA DOXYGEN COVERAGE - cpp/src/")
    print("=" * 80)
    
    cpp_dir = 'cpp/src'
    cpp_stats = {
        'files': 0,
        'missing_class': 0,
        'missing_func': 0,
        'total_class': 0,
        'total_func': 0,
        'file_details': []
    }
    
    for f in os.listdir(cpp_dir):
        if f.endswith(('.cpp', '.cu', '.hpp', '.h')):
            filepath = os.path.join(cpp_dir, f)
            result = analyze_cpp_file(filepath)
            if result:
                cpp_stats['files'] += 1
                
                # Separate classes from functions
                for item in result.items:
                    # Heuristic: classes are typically CapWord, functions are camelCase or snake_case
                    if re.match(r'^[A-Z][a-zA-Z0-9]*$', item.name):
                        cpp_stats['total_class'] += 1
                        if not item.has_doc:
                            cpp_stats['missing_class'] += 1
                    else:
                        cpp_stats['total_func'] += 1
                        if not item.has_doc:
                            cpp_stats['missing_func'] += 1
                
                if result.missing_count > 0:
                    cpp_stats['file_details'].append(result)
    
    cpp_total = cpp_stats['total_class'] + cpp_stats['total_func']
    cpp_missing = cpp_stats['missing_class'] + cpp_stats['missing_func']
    cpp_coverage = ((cpp_total - cpp_missing) / cpp_total * 100) if cpp_total > 0 else 0
    
    print(f"\nFiles analyzed: {cpp_stats['files']}")
    print(f"Class docs missing: {cpp_stats['missing_class']}/{cpp_stats['total_class']}")
    print(f"Function docs missing: {cpp_stats['missing_func']}/{cpp_stats['total_func']}")
    print(f"\nOverall C++/CUDA coverage: {cpp_coverage:.1f}%")
    
    # Show C++ files needing attention
    print("\n" + "-" * 80)
    print("C++/CUDA FILES NEEDING ATTENTION")
    print("-" * 80)
    cpp_stats['file_details'].sort(key=lambda x: x.missing_count, reverse=True)
    for item in cpp_stats['file_details']:
        total = len(item.items)
        missing = item.missing_count
        cov = ((total - missing) / total * 100) if total > 0 else 0
        header = "✓ has @file or /** */ header" if item.has_header else "✗ no header doc"
        print(f"\n  [{header}] {item.path}")
        print(f"    {missing}/{total} items missing ({cov:.0f}% coverage)")
        # Show first 15 undocumented items
        undocumented = [i for i in item.items if not i.has_doc][:15]
        for i in undocumented:
            print(f"      - {i.name} (line {i.line})")
        if missing > 15:
            print(f"      ... and {missing - 15} more")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPython (swe2d/):")
    print(f"  Files: {py_stats['files']}")
    print(f"  Classes: {py_stats['total_class']} (missing: {py_stats['missing_class']})")
    print(f"  Functions: {py_stats['total_func']} (missing: {py_stats['missing_func']})")
    print(f"  Coverage: {py_coverage:.1f}%")
    
    print(f"\nC++/CUDA (cpp/src/):")
    print(f"  Files: {cpp_stats['files']}")
    print(f"  Classes: {cpp_stats['total_class']} (missing: {cpp_stats['missing_class']})")
    print(f"  Functions: {cpp_stats['total_func']} (missing: {cpp_stats['missing_func']})")
    print(f"  Coverage: {cpp_coverage:.1f}%")
    
    combined_total = py_total + cpp_total
    combined_missing = py_missing + cpp_missing
    combined_coverage = ((combined_total - combined_missing) / combined_total * 100) if combined_total > 0 else 0
    
    print(f"\n{'=' * 80}")
    print(f"COMBINED COVERAGE: {combined_coverage:.1f}%")
    print(f"  Total items: {combined_total}")
    print(f"  Missing documentation: {combined_missing}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
