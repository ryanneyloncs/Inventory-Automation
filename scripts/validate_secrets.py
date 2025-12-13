#!/usr/bin/env python3
"""
Pre-deployment validation script.
Checks for placeholder values that must be changed before deploying.
"""

import sys
import os
import re

# Patterns that indicate placeholder/unsafe values
PLACEHOLDER_PATTERNS = [
    r'CHANGEME',
    r'change-this',
    r'your-.*-key',
    r'your-.*-password',
    r'your-.*-secret',
    r'placeholder',
    r'CHANGE-THIS',
    r'TODO',
    r'FIXME',
]

# Files to check
FILES_TO_CHECK = [
    'deployments/kubernetes/02-secrets.yaml',
    'deployments/kubernetes/api.yaml',
    'deployments/kubernetes/postgres.yaml',
    'docker-compose.yml',
    '.env',
]

# Known unsafe default values
UNSAFE_DEFAULTS = [
    'postgres:postgres',
    'admin:admin',
    'password:password',
    'secret:secret',
]


def check_file(filepath):
    """Check a single file for placeholder values."""
    issues = []
    
    if not os.path.exists(filepath):
        return issues  # File doesn't exist, skip
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue
            
        for pattern in PLACEHOLDER_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append({
                    'file': filepath,
                    'line': line_num,
                    'content': line.strip(),
                    'pattern': pattern
                })
                break
        
        for unsafe in UNSAFE_DEFAULTS:
            if unsafe in line.lower():
                issues.append({
                    'file': filepath,
                    'line': line_num,
                    'content': line.strip(),
                    'pattern': f'Unsafe default: {unsafe}'
                })
    
    return issues


def main():
    """Run validation checks."""
    print("=" * 60)
    print("🔒 Pre-Deployment Security Validation")
    print("=" * 60)
    print()
    
    all_issues = []
    
    # Get project root (script is in scripts/ folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    for filepath in FILES_TO_CHECK:
        issues = check_file(filepath)
        all_issues.extend(issues)
    
    if all_issues:
        print("❌ VALIDATION FAILED - Found placeholder/unsafe values:\n")
        
        for issue in all_issues:
            print(f"  File: {issue['file']}:{issue['line']}")
            print(f"  Issue: {issue['pattern']}")
            print(f"  Content: {issue['content']}")
            print()
        
        print("=" * 60)
        print(f"Total issues found: {len(all_issues)}")
        print("\n⚠️  Please replace all placeholder values before deploying!")
        print("=" * 60)
        sys.exit(1)
    else:
        print("✅ VALIDATION PASSED - No placeholder values found!")
        print()
        print("Note: This script checks for common patterns but may not catch")
        print("all security issues. Always review secrets manually.")
        print("=" * 60)
        sys.exit(0)


if __name__ == '__main__':
    main()
