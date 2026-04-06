#!/bin/bash
# Script to configure Claude Desktop for MCP server

set -e

CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

# Get the absolute path to this project directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detect uv path
UV_PATH=$(which uv)
if [ -z "$UV_PATH" ]; then
  echo "❌ Error: uv not found in PATH"
  echo "   Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "🔧 Setting up Claude Desktop MCP configuration..."
echo "📍 Project directory: $PROJECT_DIR"
echo "🔨 Using uv from: $UV_PATH"

# Create directory if it doesn't exist
mkdir -p "$CLAUDE_CONFIG_DIR"

# Create the config file with dynamic paths
cat > "$CLAUDE_CONFIG_FILE" << EOF
{
  "mcpServers": {
    "qdrant-financial-docs": {
      "command": "$UV_PATH",
      "args": [
        "--directory",
        "$PROJECT_DIR",
        "run",
        "servers/mcp_server.py"
      ] 
    }
  }
}
EOF

echo "✅ Configuration created at:"
echo "   $CLAUDE_CONFIG_FILE"
echo ""
echo "📄 Configuration contents:"
cat "$CLAUDE_CONFIG_FILE"
echo ""
echo "✨ Setup complete!"
echo ""
echo "📦 Configured MCP Server:"
echo "   • qdrant-financial-docs - Financial document search with multimodal support"
echo ""
echo "⚠️  IMPORTANT NEXT STEPS:"
echo "   1. Quit Claude Desktop completely (Cmd+Q)"
echo "   2. Reopen Claude Desktop"
echo "   3. Start a new chat and look for the 🔌 icon"
echo "   4. Try: 'What data do you have available?'"
echo ""
