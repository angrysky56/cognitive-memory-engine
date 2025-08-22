#!/usr/bin/env python3
"""
Script to fix JSON serialization issues in the CME MCP server
by replacing default=str with default=serialize_complex_objects
"""

import re
from pathlib import Path


def fix_serialization_in_file(file_path: Path):
    """Fix serialization issues in a Python file."""

    # Read the file
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    # Replace default=str with default=serialize_complex_objects
    # Use regex to match the pattern more precisely
    pattern = r'json\.dumps\([^)]*default=str([^)]*)\)'

    def replacement(match):
        # Replace default=str with default=serialize_complex_objects
        full_match = match.group(0)
        return full_match.replace('default=str', 'default=serialize_complex_objects')

    updated_content = re.sub(pattern, replacement, content)

    # Also handle cases where it's on multiple lines
    updated_content = updated_content.replace('default=str', 'default=serialize_complex_objects')

    # Write back if changes were made
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"✅ Fixed serialization in {file_path}")
        return True
    else:
        print(f"ℹ️  No changes needed in {file_path}")
        return False

def main():
    """Main function to fix serialization issues."""

    # Target file
    mcp_server_file = Path("/home/ty/Repositories/ai_workspace/cognitive-memory-engine/src/cognitive_memory_engine/mcp_server/main.py")

    if not mcp_server_file.exists():
        print(f"❌ File not found: {mcp_server_file}")
        return

    print("🔧 Fixing JSON serialization issues in CME MCP server...")

    # Fix the file
    changed = fix_serialization_in_file(mcp_server_file)

    if changed:
        print("✅ Serialization fixes applied successfully!")
        print("🧪 Test the CME server to verify the AnyURL serialization issue is resolved.")
    else:
        print("ℹ️  No changes were needed.")

if __name__ == "__main__":
    main()
