import pytest
import subprocess
import sys

def test_cli_help():
    result = subprocess.run([sys.executable, '-m', 'market_research_agent.cli', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Market Research Agent CLI" in result.stdout
