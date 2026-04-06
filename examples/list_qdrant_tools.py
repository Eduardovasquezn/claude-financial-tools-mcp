"""List all available Qdrant tools."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.tools.qdrant_tools import get_qdrant_tools

console = Console()


def list_qdrant_tools():
    """Display all available Qdrant tools with their descriptions."""
    
    tools = get_qdrant_tools()
    
    console.print(Panel.fit(
        f"[bold cyan]📦 Available Qdrant Tools ({len(tools)})[/bold cyan]\n"
        "[dim]Custom LangChain tools wrapping your Qdrant helpers[/dim]",
        border_style="cyan"
    ))
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta", border_style="blue")
    table.add_column("Tool Name", style="cyan", no_wrap=True, width=30)
    table.add_column("Description", style="white", width=60)
    table.add_column("Args", style="yellow")
    
    for tool in tools:
        # Get tool info
        name = tool.name
        description = tool.description.split('\n\n')[0] if tool.description else "No description"
        
        # Get argument names
        args = []
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema
            # Use model_fields for Pydantic V2 compatibility
            if hasattr(schema, 'model_fields'):
                args = list(schema.model_fields.keys())
            elif hasattr(schema, '__fields__'):  # Fallback for older versions
                args = list(schema.__fields__.keys())
        
        args_str = ", ".join(args) if args else "None"
        
        table.add_row(name, description, args_str)
    
    console.print(table)
    
    # Usage examples
    console.print("\n[bold]💡 Usage Examples:[/bold]\n")
    
    examples = [
        {
            "title": "Semantic Search",
            "code": 'search_documents_semantic.invoke({\n    "query": "NVIDIA data center revenue",\n    "limit": 5\n})'
        },
        {
            "title": "Search by Metadata",
            "code": 'search_by_metadata.invoke({\n    "ticker": "NVDA",\n    "quarter": "Q1",\n    "year": "2026",\n    "limit": 3\n})'
        },
        {
            "title": "Compare Quarters",
            "code": 'compare_quarters.invoke({\n    "ticker": "NVDA",\n    "quarters": ["Q1", "Q2"],\n    "year": "2026",\n    "query": "revenue growth",\n    "limit_per_quarter": 2\n})'
        },
        {
            "title": "Get Available Metadata",
            "code": 'get_available_metadata.invoke({"ticker": "NVDA"})'
        },
        {
            "title": "Use with LangGraph Agent",
            "code": """from src.tools.registry import ToolRegistry
from langgraph.prebuilt import create_react_agent

registry = ToolRegistry()
tools = registry.qdrant_tools
agent = create_react_agent(llm, tools)"""
        }
    ]
    
    for example in examples:
        console.print(f"[cyan]▸[/cyan] [bold]{example['title']}:[/bold]")
        console.print(f"  [dim]{example['code']}[/dim]\n")


if __name__ == "__main__":
    list_qdrant_tools()
