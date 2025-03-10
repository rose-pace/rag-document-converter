"""
Command-line interface for the RAG Document Converter.

This module provides a command-line interface for converting markdown documents
to RAG-optimized format, with options for individual and batch processing.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click

from rag_converter.config import load_config
from rag_converter.converter import DocumentConverter


@click.group()
@click.version_option()
def cli():
    """
    RAG Document Converter - Convert markdown documents to RAG-optimized format.
    
    This tool processes markdown documents and optimizes them for retrieval-augmented
    generation (RAG) by applying consistent formatting, entity identifiers, and
    relationship structures.
    """
    pass


@cli.command('convert')
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path())
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to custom configuration file')
@click.option('--use-llm', '-l', is_flag=True, help='Enable LLM integration for enhanced processing')
def convert_document(input_path: str, output_path: str, config: Optional[str] = None, use_llm: bool = False) -> None:
    """
    Convert a single markdown document to RAG-optimized format.
    
    Args:
        input_path: Path to the input markdown document
        output_path: Path where the optimized document will be saved
        config: Optional path to a custom configuration file
        use_llm: Whether to use LLM for enhanced processing
    """
    try:
        converter = DocumentConverter(use_llm=use_llm, config_path=config)
        success = converter.convert_document(input_path, output_path)
        
        if success:
            click.echo(f'Successfully converted document: {output_path}')
        else:
            click.echo(f'Failed to convert document: {input_path}', err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f'Error converting document: {str(e)}', err=True)
        sys.exit(1)


@cli.command('batch')
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('--pattern', '-p', default='*.md', help='Glob pattern for matching markdown files')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to custom configuration file')
@click.option('--use-llm', '-l', is_flag=True, help='Enable LLM integration for enhanced processing')
def batch_convert(input_dir: str, output_dir: str, pattern: str = '*.md', 
                 config: Optional[str] = None, use_llm: bool = False) -> None:
    """
    Convert multiple markdown documents in a directory.
    
    Args:
        input_dir: Path to directory containing input documents
        output_dir: Path where optimized documents will be saved
        pattern: Glob pattern to match markdown files (default: *.md)
        config: Optional path to a custom configuration file
        use_llm: Whether to use LLM for enhanced processing
    """
    try:
        converter = DocumentConverter(use_llm=use_llm, config_path=config)
        results = converter.batch_convert(input_dir, output_dir, file_pattern=pattern)
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        click.echo(f'Batch conversion completed: {success_count}/{total_count} files successful')
        
        if success_count < total_count:
            failed_files = [file for file, success in results.items() if not success]
            click.echo(f'Failed files: {", ".join(failed_files)}', err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f'Error during batch conversion: {str(e)}', err=True)
        sys.exit(1)


def main():
    """
    Main entry point for the command-line interface.
    
    This function is registered as a console script entry point in setup.py
    or pyproject.toml.
    
    Returns:
        The exit code from the CLI command
    """
    return cli()


if __name__ == '__main__':
    main()
