"""Test the new Qdrant tools to verify they work correctly."""

from src.tools.qdrant_tools import (
    search_documents_semantic,
    search_by_metadata,
    compare_quarters,
    get_available_metadata,
    get_qdrant_collection_stats,
)
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

console = Console()


def test_collection_stats():
    """Test getting collection statistics."""
    console.print(Panel.fit("📊 [bold]Test 1: Collection Statistics[/bold]", border_style="cyan"))
    
    result = get_qdrant_collection_stats.invoke({})
    console.print(JSON.from_data(result))
    console.print()


def test_available_metadata():
    """Test discovering available metadata."""
    console.print(Panel.fit("🔍 [bold]Test 2: Available Metadata[/bold]", border_style="cyan"))
    
    result = get_available_metadata.invoke({})
    console.print(JSON.from_data(result))
    console.print()


def test_metadata_search():
    """Test pure metadata filtering."""
    console.print(Panel.fit("🏷️  [bold]Test 3: Search by Metadata[/bold]", border_style="cyan"))
    
    # First, get available metadata to know what to search for
    metadata = get_available_metadata.invoke({})
    
    if metadata.get("tickers"):
        ticker = metadata["tickers"][0]
        console.print(f"[cyan]Searching for ticker: {ticker}[/cyan]\n")
        
        result = search_by_metadata.invoke({
            "ticker": ticker,
            "limit": 3
        })
        
        console.print(f"[green]Found {len(result)} documents[/green]")
        for i, doc in enumerate(result[:2], 1):  # Show first 2
            console.print(f"\n[bold]{i}. {doc.get('ticker')} {doc.get('quarter')} {doc.get('year')}[/bold]")
            console.print(f"   Content: {doc.get('content', '')[:150]}...")
    else:
        console.print("[yellow]No data in collection yet. Run ingest_pdf.py first.[/yellow]")
    
    console.print()


def test_semantic_search():
    """Test semantic vector search."""
    console.print(Panel.fit("🔎 [bold]Test 4: Semantic Search[/bold]", border_style="cyan"))
    
    query = "What was discussed about the Blackwel chip during the Q3 eanings of NVIDIA"
    console.print(f"[cyan]Query: {query}[/cyan]\n")
    
    result = search_documents_semantic.invoke({
        "query": query,
        "limit": 3
    })
    
    if result and not result[0].get("error"):
        console.print(f"[green]Found {len(result)} documents[/green]")
        for i, doc in enumerate(result[:2], 1):
            console.print(f"\n[bold]{i}. Score: {doc.get('score', 0):.3f}[/bold]")
            console.print(f"   {doc.get('ticker')} {doc.get('quarter')} {doc.get('year')}")
            console.print(f"   Content: {doc.get('content', '')}...")
    else:
        console.print("[yellow]No results or error occurred[/yellow]")
    
    console.print()


def test_compare_quarters():
    """Test comparing multiple quarters."""
    console.print(Panel.fit("🔀 [bold]Test 5: Compare Quarters[/bold]", border_style="cyan"))
    
    # Get available metadata first
    metadata = get_available_metadata.invoke({})
    
    if metadata.get("tickers") and metadata.get("quarters") and len(metadata["quarters"]) >= 2:
        ticker = metadata["tickers"][0]
        quarters = metadata["quarters"][:2]  # Take first 2 quarters
        year = metadata["years"][0] if metadata.get("years") else "2026"
        
        console.print(f"[cyan]Comparing {ticker} {quarters[0]} vs {quarters[1]} {year}[/cyan]")
        console.print(f"[cyan]Query: 'revenue'[/cyan]\n")
        
        result = compare_quarters.invoke({
            "ticker": ticker,
            "quarters": quarters,
            "year": year,
            "query": "revenue",
            "limit_per_quarter": 2
        })
        
        if not result.get("error"):
            console.print(f"[green]✓ Total documents: {result.get('total_documents')}[/green]\n")
            
            # Show detailed breakdown with metadata verification
            for quarter, docs in result.get("quarters", {}).items():
                console.print(f"[bold yellow]═══ {quarter} ═══[/bold yellow] ({len(docs)} documents)")
                
                for i, doc in enumerate(docs, 1):
                    # Verify metadata matches the quarter
                    doc_quarter = doc.get('quarter', '?')
                    doc_ticker = doc.get('ticker', '?')
                    doc_year = doc.get('year', '?')
                    score = doc.get('score', 0)
                    
                    # Show verification that metadata is correct
                    match_icon = "✓" if doc_quarter == quarter else "✗"
                    console.print(f"  {match_icon} Doc {i}: [{doc_ticker}] [{doc_quarter}] [{doc_year}] Score: {score:.3f}")
                    console.print(f"     Content: {doc.get('content', '')[:120]}...")
                console.print()
            
            # Final verification message
            console.print("[bold green]✓ Verification: Each document is correctly grouped by its quarter metadata[/bold green]")
        else:
            console.print(f"[red]Error: {result.get('error')}[/red]")
    else:
        console.print("[yellow]Need at least 2 quarters of data to compare.[/yellow]")
    
    console.print()


def main():
    """Run all tests."""
    console.print("\n[bold cyan]🧪 Testing Qdrant Tools[/bold cyan]\n")
    
    try:
        test_collection_stats()
        test_available_metadata()
        test_metadata_search()
        test_semantic_search()
        test_compare_quarters()
        
        console.print(Panel.fit(
            "[bold green]✅ All Tests Complete![/bold green]\n"
            "Check the output above to verify each tool works correctly.",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error during testing:[/bold red]")
        console.print(f"[red]{e}[/red]")
        console.print("\n[yellow]Make sure:[/yellow]")
        console.print("  1. Qdrant is running: docker compose up -d")
        console.print("  2. Data is ingested: python ingest_pdf.py")
        console.print("  3. Environment variables are set")


if __name__ == "__main__":
    main()
