#!/bin/bash
# Script to configure Claude Desktop for MCP server

set -e

CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

echo "🔧 Setting up Claude Desktop MCP configuration..."

# Create directory if it doesn't exist
mkdir -p "$CLAUDE_CONFIG_DIR"

# Create the config file
cat > "$CLAUDE_CONFIG_FILE" << 'EOF'
{
  "mcpServers": {
    "qdrant-financial-docs": {
      "command": "/Users/eduardo.vasquez.nolasco/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/eduardo.vasquez.nolasco/Documents/testing/agent-tools-claude",
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
echo "📦 Configured MCP Servers:"
echo "   1. financial-agent - Sentiment analysis (Reddit + Twitter/X)"
echo "   2. qdrant-financial-docs - Document search (Qdrant)"
echo ""
echo "⚠️  IMPORTANT NEXT STEPS:"
echo "   1. Quit Claude Desktop completely (Cmd+Q)"
echo "   2. Reopen Claude Desktop"
echo "   3. Start a new chat and look for the 🔌 icon"
echo "   4. Try: 'What tickers do you have financial data for?'"
echo ""
