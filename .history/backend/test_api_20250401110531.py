"""
Test script for the Filipino Dictionary GraphQL API.
"""

import requests
import json
from typing import Dict, Any, List
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GraphQL endpoint
API_URL = "http://localhost:5000/graphql"

# Rich console for pretty output
console = Console()

def execute_query(query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute a GraphQL query and return the response."""
    try:
        response = requests.post(
            API_URL,
            json={
                'query': query,
                'variables': variables or {}
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Query failed: {e}")
        return {'errors': [{'message': str(e)}]}

def test_word_query():
    """Test basic word query."""
    query = """
    query {
        word(lemma: "aklat") {
            id
            lemma
            normalizedLemma
            languageCode
            hasBaybayin
            baybayinForm
            definitions {
                definitionText
                originalPos
            }
        }
    }
    """
    
    result = execute_query(query)
    console.print("\n[bold]Testing Word Query:[/bold]")
    if 'errors' in result:
        console.print("[red]❌ Word query failed:[/red]")
        console.print(result['errors'])
    else:
        console.print("[green]✓ Word query successful[/green]")
        console.print(json.dumps(result, indent=2))

def test_search_words():
    """Test word search functionality."""
    query = """
    query($searchTerm: String!) {
        searchWords(query: $searchTerm, mode: "all", limit: 5) {
            lemma
            normalizedLemma
            languageCode
            hasBaybayin
        }
    }
    """
    
    variables = {'searchTerm': 'baha'}
    result = execute_query(query, variables)
    
    console.print("\n[bold]Testing Word Search:[/bold]")
    if 'errors' in result:
        console.print("[red]❌ Search query failed:[/red]")
        console.print(result['errors'])
    else:
        console.print("[green]✓ Search query successful[/green]")
        
        # Create table for results
        table = Table(show_header=True, header_style="bold")
        table.add_column("Lemma")
        table.add_column("Normalized")
        table.add_column("Language")
        table.add_column("Has Baybayin")
        
        for word in result.get('data', {}).get('searchWords', []):
            table.add_row(
                word['lemma'],
                word['normalizedLemma'],
                word['languageCode'],
                '✓' if word['hasBaybayin'] else '✗'
            )
        
        console.print(table)

def test_find_related_words():
    """Test related words functionality."""
    query = """
    query {
        findRelatedWords(wordId: "1", relationType: "synonym", limit: 5) {
            lemma
            normalizedLemma
            languageCode
        }
    }
    """
    
    result = execute_query(query)
    console.print("\n[bold]Testing Related Words:[/bold]")
    if 'errors' in result:
        console.print("[red]❌ Related words query failed:[/red]")
        console.print(result['errors'])
    else:
        console.print("[green]✓ Related words query successful[/green]")
        console.print(json.dumps(result, indent=2))

def test_word_forms():
    """Test word forms functionality."""
    query = """
    query {
        findWordForms(wordId: "1", limit: 5) {
            form
            formType
            metadata
        }
    }
    """
    
    result = execute_query(query)
    console.print("\n[bold]Testing Word Forms:[/bold]")
    if 'errors' in result:
        console.print("[red]❌ Word forms query failed:[/red]")
        console.print(result['errors'])
    else:
        console.print("[green]✓ Word forms query successful[/green]")
        console.print(json.dumps(result, indent=2))

def test_word_templates():
    """Test word templates functionality."""
    query = """
    query {
        findWordTemplates(wordId: "1", limit: 5) {
            template
            templateType
            pattern
        }
    }
    """
    
    result = execute_query(query)
    console.print("\n[bold]Testing Word Templates:[/bold]")
    if 'errors' in result:
        console.print("[red]❌ Word templates query failed:[/red]")
        console.print(result['errors'])
    else:
        console.print("[green]✓ Word templates query successful[/green]")
        console.print(json.dumps(result, indent=2))

def test_affixations():
    """Test affixations functionality."""
    query = """
    query {
        findAffixations(wordId: "1", limit: 5) {
            affixType
            sources
        }
    }
    """
    
    result = execute_query(query)
    console.print("\n[bold]Testing Affixations:[/bold]")
    if 'errors' in result:
        console.print("[red]❌ Affixations query failed:[/red]")
        console.print(result['errors'])
    else:
        console.print("[green]✓ Affixations query successful[/green]")
        console.print(json.dumps(result, indent=2))

def test_caching():
    """Test caching functionality."""
    console.print("\n[bold]Testing Cache:[/bold]")
    
    # Get initial cache stats
    query = """
    query {
        cacheStats
    }
    """
    
    initial_stats = execute_query(query)
    if 'errors' in initial_stats:
        console.print("[red]❌ Cache stats query failed:[/red]")
        console.print(initial_stats['errors'])
        return
    
    console.print("[green]✓ Cache stats query successful[/green]")
    console.print("Initial cache stats:")
    console.print(json.dumps(initial_stats['data']['cacheStats'], indent=2))
    
    # Test word query caching
    word_query = """
    query {
        word(lemma: "aklat") {
            word {
                lemma
                normalizedLemma
            }
            success
        }
    }
    """
    
    # First request (cache miss)
    start_time = time.time()
    result1 = execute_query(word_query)
    first_request_time = time.time() - start_time
    
    # Second request (cache hit)
    start_time = time.time()
    result2 = execute_query(word_query)
    second_request_time = time.time() - start_time
    
    if result1 == result2:
        console.print("[green]✓ Cache consistency verified[/green]")
        console.print(f"First request time: {first_request_time:.3f}s")
        console.print(f"Second request time: {second_request_time:.3f}s")
        console.print(f"Speed improvement: {(first_request_time/second_request_time):.1f}x")
    else:
        console.print("[red]❌ Cache consistency check failed[/red]")
    
    # Get final cache stats
    final_stats = execute_query(query)
    if 'data' in final_stats:
        console.print("\nFinal cache stats:")
        console.print(json.dumps(final_stats['data']['cacheStats'], indent=2))

def run_all_tests():
    """Run all API tests."""
    console.print("[bold blue]Running API Tests...[/bold blue]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Testing...", total=7)  # Updated total
        
        # Run tests
        test_word_query()
        progress.update(task, advance=1)
        
        test_search_words()
        progress.update(task, advance=1)
        
        test_find_related_words()
        progress.update(task, advance=1)
        
        test_word_forms()
        progress.update(task, advance=1)
        
        test_word_templates()
        progress.update(task, advance=1)
        
        test_affixations()
        progress.update(task, advance=1)
        
        test_caching()  # Added cache testing
        progress.update(task, advance=1)
    
    console.print("\n[bold green]✓ All tests completed[/bold green]")

if __name__ == '__main__':
    run_all_tests() 