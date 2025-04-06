#!/usr/bin/env python
"""
Dictionary Quality Monitoring Script

This script regularly checks dictionary quality using the quality_assessment API endpoint,
tracks changes over time, and generates reports to help improve data quality.

Usage:
    python monitor_dictionary_quality.py [--output-dir DIR] [--api-url URL] [--api-key KEY]

Options:
    --output-dir DIR      Directory to store reports (default: ./quality_reports)
    --api-url URL         Base URL of the API (default: http://localhost:5000/api/v2)
    --api-key KEY         API key for authentication
    --language LANG       Filter by language code (default: all languages)
    --generate-report     Generate a detailed HTML report
    --compare             Compare with previous assessment
    --notify              Send notification if quality decreases
"""

import os
import sys
import argparse
import requests
import json
import datetime
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from jinja2 import Template
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import configparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dictionary_quality.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dictionary_quality")

# Default configuration
DEFAULT_CONFIG = {
    'api': {
        'base_url': 'http://localhost:5000/api/v2',
        'api_key': '',
        'timeout': '300',
    },
    'monitoring': {
        'output_dir': './quality_reports',
        'history_file': 'quality_history.json',
        'report_template': 'report_template.html',
        'quality_threshold': '0.7',  # Minimum acceptable average completeness
        'critical_threshold': '10',  # Maximum acceptable critical issues percentage
    },
    'notification': {
        'enabled': 'false',
        'smtp_server': 'smtp.example.com',
        'smtp_port': '587',
        'smtp_user': '',
        'smtp_password': '',
        'from_email': 'dictionary-monitor@example.com',
        'to_email': 'admin@example.com',
        'notify_on_decrease': 'true',
        'notify_threshold': '0.05',  # 5% decrease in quality
    }
}

def load_config(config_path='monitor_config.ini'):
    """Load configuration from file, create default if not exists."""
    config = configparser.ConfigParser()
    
    # Set default config
    for section, options in DEFAULT_CONFIG.items():
        if not config.has_section(section):
            config.add_section(section)
        for option, value in options.items():
            config.set(section, option, value)
    
    # Try to load from file
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        # Save default config
        with open(config_path, 'w') as f:
            config.write(f)
        logger.info(f"Created default configuration at {config_path}")
    
    return config

def get_quality_assessment(api_url, api_key, language=None, timeout=300):
    """Fetch quality assessment data from the API."""
    endpoint = f"{api_url.rstrip('/')}/quality_assessment"
    
    headers = {
        'X-Api-Key': api_key,
        'Accept': 'application/json',
    }
    
    params = {}
    if language:
        params['language_code'] = language
    
    try:
        logger.info(f"Fetching quality assessment data from {endpoint}")
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching quality assessment: {str(e)}")
        return None

def load_quality_history(history_file):
    """Load quality assessment history from file."""
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in history file: {history_file}")
            return {"assessments": []}
    return {"assessments": []}

def save_quality_history(history_file, history):
    """Save quality assessment history to file."""
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

def add_assessment_to_history(history, assessment):
    """Add current assessment to history with summary metrics."""
    # Extract key metrics for historical tracking
    assessment_summary = {
        "date": assessment["assessment_date"],
        "total_entries": assessment["total_entries"],
        "verified_percent": assessment["verification_status"]["percent_verified"],
        "average_completeness": assessment["completeness"]["average_score"],
        "completeness_distribution": assessment["completeness"]["distribution"],
        "issues": {
            "total": assessment["issue_counts"]["total"],
            "by_severity": assessment["issue_counts"]["by_severity"]
        },
        "components": {
            "with_definitions_percent": (assessment["components"]["with_definitions"] / assessment["total_entries"]) * 100 if assessment["total_entries"] > 0 else 0,
            "with_etymology_percent": (assessment["components"]["with_etymology"] / assessment["total_entries"]) * 100 if assessment["total_entries"] > 0 else 0,
            "with_pronunciations_percent": (assessment["components"]["with_pronunciations"] / assessment["total_entries"]) * 100 if assessment["total_entries"] > 0 else 0,
            "with_baybayin_percent": (assessment["components"]["with_baybayin"] / assessment["total_entries"]) * 100 if assessment["total_entries"] > 0 else 0
        }
    }
    
    # Calculate quality score (weighted average of metrics)
    quality_score = (
        assessment_summary["verified_percent"] * 0.2 +
        assessment_summary["average_completeness"] * 100 * 0.4 +
        assessment_summary["components"]["with_definitions_percent"] * 0.2 +
        (assessment_summary["components"]["with_etymology_percent"] +
         assessment_summary["components"]["with_pronunciations_percent"] +
         assessment_summary["components"]["with_baybayin_percent"]) / 3 * 0.2
    ) / 100.0
    
    # Add to summary
    assessment_summary["quality_score"] = round(quality_score, 2)
    
    # Add to history
    history["assessments"].append(assessment_summary)
    
    return history

def compare_with_previous(current, history):
    """Compare current assessment with previous."""
    if not history["assessments"]:
        return {
            "is_first_assessment": True,
            "changes": {}
        }
    
    # Get the previous assessment
    previous = history["assessments"][-1]
    
    # Calculate changes
    changes = {
        "total_entries": {
            "previous": previous["total_entries"],
            "current": current["total_entries"],
            "change": current["total_entries"] - previous["total_entries"],
            "percent_change": round(((current["total_entries"] - previous["total_entries"]) / previous["total_entries"]) * 100, 1) if previous["total_entries"] > 0 else None
        },
        "verified_percent": {
            "previous": previous["verified_percent"],
            "current": current["verified_percent"],
            "change": round(current["verified_percent"] - previous["verified_percent"], 1),
        },
        "average_completeness": {
            "previous": previous["average_completeness"],
            "current": current["average_completeness"],
            "change": round(current["average_completeness"] - previous["average_completeness"], 2),
        },
        "completeness_distribution": {
            "previous": previous["completeness_distribution"],
            "current": current["completeness_distribution"],
            "change": {
                category: current["completeness_distribution"].get(category, 0) - previous["completeness_distribution"].get(category, 0)
                for category in set(current["completeness_distribution"]).union(previous["completeness_distribution"])
            }
        },
        "issues": {
            "previous": previous["issues"]["total"],
            "current": current["issues"]["total"],
            "change": current["issues"]["total"] - previous["issues"]["total"],
        },
        "components": {
            comp: {
                "previous": previous["components"].get(comp, 0),
                "current": current["components"].get(comp, 0),
                "change": round(current["components"].get(comp, 0) - previous["components"].get(comp, 0), 1),
            }
            for comp in ["with_definitions_percent", "with_etymology_percent", "with_pronunciations_percent", "with_baybayin_percent"]
        },
        "quality_score": {
            "previous": previous["quality_score"],
            "current": current["quality_score"],
            "change": round(current["quality_score"] - previous["quality_score"], 2),
            "percent_change": round(((current["quality_score"] - previous["quality_score"]) / previous["quality_score"]) * 100, 1) if previous["quality_score"] > 0 else None
        }
    }
    
    # Is quality improving or declining?
    changes["quality_trend"] = "improving" if changes["quality_score"]["change"] > 0 else "declining" if changes["quality_score"]["change"] < 0 else "stable"
    
    return {
        "is_first_assessment": False,
        "days_since_previous": (datetime.datetime.fromisoformat(current["date"]) - datetime.datetime.fromisoformat(previous["date"])).days,
        "changes": changes
    }

def generate_charts(history, output_dir):
    """Generate charts from quality history data."""
    if not history["assessments"]:
        logger.warning("No history data to generate charts from")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    charts = []
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([
        {
            "date": datetime.datetime.fromisoformat(assessment["date"]).date(),
            "quality_score": assessment["quality_score"],
            "verified_percent": assessment["verified_percent"],
            "average_completeness": assessment["average_completeness"],
            "with_definitions": assessment["components"]["with_definitions_percent"],
            "with_etymology": assessment["components"]["with_etymology_percent"],
            "with_pronunciations": assessment["components"]["with_pronunciations_percent"],
            "with_baybayin": assessment["components"]["with_baybayin_percent"],
            "total_entries": assessment["total_entries"],
            "critical_issues": assessment["issues"]["by_severity"].get("critical", 0),
            "warning_issues": assessment["issues"]["by_severity"].get("warning", 0),
            "info_issues": assessment["issues"]["by_severity"].get("info", 0)
        }
        for assessment in history["assessments"]
    ])
    
    # Ensure there are at least 2 data points
    if len(df) < 2:
        # Add a copy of the first point with an earlier date for basic trending
        first_row = df.iloc[0].copy()
        first_row['date'] = first_row['date'] - datetime.timedelta(days=7)
        df = pd.concat([pd.DataFrame([first_row]), df], ignore_index=True)
    
    # 1. Overall Quality Score Trend
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['quality_score'], marker='o', linestyle='-', color='#3366cc')
    plt.title('Dictionary Quality Score Trend')
    plt.xlabel('Date')
    plt.ylabel('Quality Score (0-1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    
    # Add trend line
    z = np.polyfit(range(len(df)), df['quality_score'], 1)
    p = np.poly1d(z)
    plt.plot(df['date'], p(range(len(df))), "r--", alpha=0.8)
    
    quality_trend_chart = os.path.join(output_dir, 'quality_trend.png')
    plt.savefig(quality_trend_chart)
    plt.close()
    charts.append(('Quality Score Trend', 'quality_trend.png'))
    
    # 2. Component Completion Rates
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['with_definitions'], marker='o', label='Definitions')
    plt.plot(df['date'], df['with_etymology'], marker='s', label='Etymology')
    plt.plot(df['date'], df['with_pronunciations'], marker='^', label='Pronunciation')
    plt.plot(df['date'], df['with_baybayin'], marker='x', label='Baybayin')
    plt.title('Component Completion Rates')
    plt.xlabel('Date')
    plt.ylabel('Completion Percentage')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    
    components_chart = os.path.join(output_dir, 'components_trend.png')
    plt.savefig(components_chart)
    plt.close()
    charts.append(('Component Completion Rates', 'components_trend.png'))
    
    # 3. Issues by Severity
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['critical_issues'], marker='o', color='red', label='Critical')
    plt.plot(df['date'], df['warning_issues'], marker='s', color='orange', label='Warning')
    plt.plot(df['date'], df['info_issues'], marker='^', color='blue', label='Info')
    plt.title('Issues by Severity')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    issues_chart = os.path.join(output_dir, 'issues_trend.png')
    plt.savefig(issues_chart)
    plt.close()
    charts.append(('Issues by Severity', 'issues_trend.png'))
    
    # 4. Dictionary Growth
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['total_entries'], marker='o', linestyle='-', color='green')
    plt.title('Dictionary Growth')
    plt.xlabel('Date')
    plt.ylabel('Total Entries')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    growth_chart = os.path.join(output_dir, 'growth_trend.png')
    plt.savefig(growth_chart)
    plt.close()
    charts.append(('Dictionary Growth', 'growth_trend.png'))
    
    return charts

def generate_html_report(assessment, comparison, charts, output_dir, config):
    """Generate HTML report with assessment results and charts."""
    report_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dictionary Quality Report - {{ date }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .header {
                background-color: #3498db;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .summary {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .change {
                font-size: 14px;
                padding: 3px 6px;
                border-radius: 3px;
            }
            .positive-change {
                background-color: #d4edda;
                color: #155724;
            }
            .negative-change {
                background-color: #f8d7da;
                color: #721c24;
            }
            .neutral-change {
                background-color: #e2e3e5;
                color: #383d41;
            }
            .charts {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }
            .chart {
                flex: 1 1 45%;
                min-width: 400px;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
            .chart img {
                width: 100%;
                height: auto;
            }
            .issues {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 30px;
            }
            .issue-item {
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 3px;
            }
            .critical {
                background-color: #f8d7da;
                border-left: 4px solid #dc3545;
            }
            .warning {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
            }
            .info {
                background-color: #d1ecf1;
                border-left: 4px solid #17a2b8;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .footer {
                margin-top: 40px;
                text-align: center;
                font-size: 12px;
                color: #6c757d;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Dictionary Quality Report</h1>
            <p>Generated on {{ date }}</p>
        </div>
        
        <h2>Quality Summary</h2>
        <div class="summary">
            <div class="metric-card">
                <h3>Overall Quality Score</h3>
                <div class="metric-value">{{ assessment.quality_score }}</div>
                {% if not comparison.is_first_assessment %}
                <div class="change {{ 'positive-change' if comparison.changes.quality_score.change > 0 else 'negative-change' if comparison.changes.quality_score.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.quality_score.change | abs | round(2) }} {{ "▲" if comparison.changes.quality_score.change > 0 else "▼" if comparison.changes.quality_score.change < 0 else "━" }}
                    ({{ comparison.changes.quality_score.percent_change | abs | round(1) }}%)
                </div>
                {% endif %}
                <p>Quality threshold: {{ quality_threshold }}</p>
            </div>
            
            <div class="metric-card">
                <h3>Dictionary Size</h3>
                <div class="metric-value">{{ assessment.total_entries }}</div>
                {% if not comparison.is_first_assessment and comparison.changes.total_entries.change != 0 %}
                <div class="change {{ 'positive-change' if comparison.changes.total_entries.change > 0 else 'negative-change' if comparison.changes.total_entries.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.total_entries.change | abs }} {{ "▲" if comparison.changes.total_entries.change > 0 else "▼" if comparison.changes.total_entries.change < 0 else "━" }}
                    {% if comparison.changes.total_entries.percent_change %}
                    ({{ comparison.changes.total_entries.percent_change | abs | round(1) }}%)
                    {% endif %}
                </div>
                {% endif %}
            </div>
            
            <div class="metric-card">
                <h3>Verified Words</h3>
                <div class="metric-value">{{ assessment.verified_percent }}%</div>
                {% if not comparison.is_first_assessment %}
                <div class="change {{ 'positive-change' if comparison.changes.verified_percent.change > 0 else 'negative-change' if comparison.changes.verified_percent.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.verified_percent.change | abs | round(1) }}% {{ "▲" if comparison.changes.verified_percent.change > 0 else "▼" if comparison.changes.verified_percent.change < 0 else "━" }}
                </div>
                {% endif %}
            </div>
            
            <div class="metric-card">
                <h3>Average Completeness</h3>
                <div class="metric-value">{{ assessment.average_completeness | round(2) }}</div>
                {% if not comparison.is_first_assessment %}
                <div class="change {{ 'positive-change' if comparison.changes.average_completeness.change > 0 else 'negative-change' if comparison.changes.average_completeness.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.average_completeness.change | abs | round(2) }} {{ "▲" if comparison.changes.average_completeness.change > 0 else "▼" if comparison.changes.average_completeness.change < 0 else "━" }}
                </div>
                {% endif %}
            </div>
        </div>
        
        {% if charts %}
        <h2>Trends</h2>
        <div class="charts">
            {% for chart_title, chart_file in charts %}
            <div class="chart">
                <h3>{{ chart_title }}</h3>
                <img src="{{ chart_file }}" alt="{{ chart_title }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <h2>Component Completion</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Completion Rate</th>
                {% if not comparison.is_first_assessment %}
                <th>Change</th>
                {% endif %}
            </tr>
            <tr>
                <td>Definitions</td>
                <td>{{ assessment.components.with_definitions_percent | round(1) }}%</td>
                {% if not comparison.is_first_assessment %}
                <td class="{{ 'positive-change' if comparison.changes.components.with_definitions_percent.change > 0 else 'negative-change' if comparison.changes.components.with_definitions_percent.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.components.with_definitions_percent.change | abs | round(1) }}% {{ "▲" if comparison.changes.components.with_definitions_percent.change > 0 else "▼" if comparison.changes.components.with_definitions_percent.change < 0 else "━" }}
                </td>
                {% endif %}
            </tr>
            <tr>
                <td>Etymology</td>
                <td>{{ assessment.components.with_etymology_percent | round(1) }}%</td>
                {% if not comparison.is_first_assessment %}
                <td class="{{ 'positive-change' if comparison.changes.components.with_etymology_percent.change > 0 else 'negative-change' if comparison.changes.components.with_etymology_percent.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.components.with_etymology_percent.change | abs | round(1) }}% {{ "▲" if comparison.changes.components.with_etymology_percent.change > 0 else "▼" if comparison.changes.components.with_etymology_percent.change < 0 else "━" }}
                </td>
                {% endif %}
            </tr>
            <tr>
                <td>Pronunciations</td>
                <td>{{ assessment.components.with_pronunciations_percent | round(1) }}%</td>
                {% if not comparison.is_first_assessment %}
                <td class="{{ 'positive-change' if comparison.changes.components.with_pronunciations_percent.change > 0 else 'negative-change' if comparison.changes.components.with_pronunciations_percent.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.components.with_pronunciations_percent.change | abs | round(1) }}% {{ "▲" if comparison.changes.components.with_pronunciations_percent.change > 0 else "▼" if comparison.changes.components.with_pronunciations_percent.change < 0 else "━" }}
                </td>
                {% endif %}
            </tr>
            <tr>
                <td>Baybayin</td>
                <td>{{ assessment.components.with_baybayin_percent | round(1) }}%</td>
                {% if not comparison.is_first_assessment %}
                <td class="{{ 'positive-change' if comparison.changes.components.with_baybayin_percent.change > 0 else 'negative-change' if comparison.changes.components.with_baybayin_percent.change < 0 else 'neutral-change' }}">
                    {{ comparison.changes.components.with_baybayin_percent.change | abs | round(1) }}% {{ "▲" if comparison.changes.components.with_baybayin_percent.change > 0 else "▼" if comparison.changes.components.with_baybayin_percent.change < 0 else "━" }}
                </td>
                {% endif %}
            </tr>
        </table>
        
        <h2>Issues Summary</h2>
        <div class="issues">
            <h3>Total Issues: {{ full_assessment.issue_counts.total }}</h3>
            <p>
                Critical: {{ full_assessment.issue_counts.by_severity.critical }} |
                Warning: {{ full_assessment.issue_counts.by_severity.warning }} |
                Info: {{ full_assessment.issue_counts.by_severity.info }}
            </p>
            
            {% if full_assessment.issues %}
            <h3>Top Issues</h3>
            {% for issue in full_assessment.issues %}
            <div class="issue-item {{ issue.severity }}">
                <strong>{{ issue.message }}</strong> ({{ issue.severity }})
                <p>Suggestions: {{ issue.suggestions | join(", ") }}</p>
            </div>
            {% endfor %}
            {% endif %}
        </div>
        
        {% if not comparison.is_first_assessment %}
        <h2>Change Since Last Assessment</h2>
        <p>This report compares data from {{ comparison.days_since_previous }} days ago.</p>
        
        <div class="summary">
            <div class="metric-card">
                <h3>Quality Trend</h3>
                <div class="metric-value {{ 'positive-change' if comparison.changes.quality_trend == 'improving' else 'negative-change' if comparison.changes.quality_trend == 'declining' else 'neutral-change' }}">
                    {{ comparison.changes.quality_trend | capitalize }}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>New Entries</h3>
                <div class="metric-value">{{ comparison.changes.total_entries.change }}</div>
            </div>
            
            <div class="metric-card">
                <h3>Issues Change</h3>
                <div class="metric-value {{ 'positive-change' if comparison.changes.issues.change < 0 else 'negative-change' if comparison.changes.issues.change > 0 else 'neutral-change' }}">
                    {{ comparison.changes.issues.change | abs }} {{ "▼" if comparison.changes.issues.change < 0 else "▲" if comparison.changes.issues.change > 0 else "━" }}
                </div>
            </div>
        </div>
        {% endif %}
        
        <h2>Completeness Distribution</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Count</th>
                <th>Percentage</th>
                {% if not comparison.is_first_assessment %}
                <th>Change</th>
                {% endif %}
            </tr>
            {% for category in ['excellent', 'good', 'fair', 'poor', 'incomplete'] %}
            <tr>
                <td>{{ category | capitalize }}</td>
                <td>{{ full_assessment.completeness.distribution[category] }}</td>
                <td>{{ full_assessment.completeness.percent_by_category[category] }}%</td>
                {% if not comparison.is_first_assessment %}
                <td class="{{ 'positive-change' if (category in ['excellent', 'good', 'fair'] and comparison.changes.completeness_distribution.change[category] > 0) or (category in ['poor', 'incomplete'] and comparison.changes.completeness_distribution.change[category] < 0) else 'negative-change' if (category in ['excellent', 'good', 'fair'] and comparison.changes.completeness_distribution.change[category] < 0) or (category in ['poor', 'incomplete'] and comparison.changes.completeness_distribution.change[category] > 0) else 'neutral-change' }}">
                    {{ comparison.changes.completeness_distribution.change[category] | abs }} {{ "▲" if comparison.changes.completeness_distribution.change[category] > 0 else "▼" if comparison.changes.completeness_distribution.change[category] < 0 else "━" }}
                </td>
                {% endif %}
            </tr>
            {% endfor %}
        </table>
        
        <div class="footer">
            <p>Generated by Dictionary Quality Monitoring Tool</p>
            <p>Report date: {{ date }}</p>
        </div>
    </body>
    </html>
    """
    
    # Prepare template variables
    template_vars = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "assessment": assessment,
        "comparison": comparison,
        "charts": charts,
        "full_assessment": assessment,
        "quality_threshold": config.get('monitoring', 'quality_threshold')
    }
    
    # Create template and render
    template = Template(report_template)
    report_html = template.render(**template_vars)
    
    # Write to file
    report_filename = os.path.join(output_dir, f"quality_report_{datetime.datetime.now().strftime('%Y%m%d')}.html")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_html)
        
    logger.info(f"Generated HTML report: {report_filename}")
    return report_filename

def send_notification(config, report_file, assessment, comparison):
    """Send email notification with report."""
    if not config.getboolean('notification', 'enabled'):
        logger.info("Email notifications are disabled")
        return False
        
    # Check if we should notify
    quality_decreased = not comparison["is_first_assessment"] and comparison["changes"]["quality_score"]["change"] < 0
    notify_threshold = float(config.get('notification', 'notify_threshold'))
    significant_decrease = comparison["is_first_assessment"] or abs(comparison["changes"]["quality_score"]["percent_change"] or 0) >= notify_threshold * 100
    
    if config.getboolean('notification', 'notify_on_decrease') and not quality_decreased:
        logger.info("Quality hasn't decreased, skipping notification")
        return False
        
    if quality_decreased and not significant_decrease:
        logger.info(f"Quality decrease ({comparison['changes']['quality_score']['percent_change']}%) is below notification threshold ({notify_threshold * 100}%)")
        return False
    
    try:
        # Prepare email
        msg = MIMEMultipart()
        msg['From'] = config.get('notification', 'from_email')
        msg['To'] = config.get('notification', 'to_email')
        
        if quality_decreased:
            msg['Subject'] = f"ALERT: Dictionary Quality Score Decreased to {assessment['quality_score']}"
            email_body = f"""
            <html>
            <body>
                <h2>Dictionary Quality Alert</h2>
                <p>The dictionary quality score has decreased from {comparison['changes']['quality_score']['previous']} to {assessment['quality_score']} ({comparison['changes']['quality_score']['percent_change']}%).</p>
                <p>This decrease is significant and requires attention.</p>
                <p>Please see the attached report for details.</p>
            </body>
            </html>
            """
        else:
            msg['Subject'] = f"Dictionary Quality Report - {datetime.datetime.now().strftime('%Y-%m-%d')}"
            email_body = f"""
            <html>
            <body>
                <h2>Dictionary Quality Report</h2>
                <p>Current quality score: {assessment['quality_score']}</p>
                <p>Please see the attached report for details.</p>
            </body>
            </html>
            """
        
        # Add body
        msg.attach(MIMEText(email_body, 'html'))
        
        # Add attachment
        with open(report_file, 'r', encoding='utf-8') as f:
            attachment = MIMEText(f.read(), 'html')
            attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_file))
            msg.attach(attachment)
        
        # Connect to SMTP server and send
        server = smtplib.SMTP(config.get('notification', 'smtp_server'), config.getint('notification', 'smtp_port'))
        server.starttls()
        
        if config.get('notification', 'smtp_user') and config.get('notification', 'smtp_password'):
            server.login(config.get('notification', 'smtp_user'), config.get('notification', 'smtp_password'))
            
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Sent notification email to {config.get('notification', 'to_email')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send notification email: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Dictionary Quality Monitoring Tool')
    parser.add_argument('--output-dir', help='Directory to store reports')
    parser.add_argument('--api-url', help='Base URL of the API')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--language', help='Filter by language code')
    parser.add_argument('--generate-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--compare', action='store_true', help='Compare with previous assessment')
    parser.add_argument('--notify', action='store_true', help='Send notification')
    parser.add_argument('--config', default='monitor_config.ini', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line args if provided
    if args.output_dir:
        config.set('monitoring', 'output_dir', args.output_dir)
    if args.api_url:
        config.set('api', 'base_url', args.api_url)
    if args.api_key:
        config.set('api', 'api_key', args.api_key)
    if args.notify:
        config.set('notification', 'enabled', 'true')
    
    # Create output directory if it doesn't exist
    output_dir = config.get('monitoring', 'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get quality assessment from API
    assessment_data = get_quality_assessment(
        config.get('api', 'base_url'),
        config.get('api', 'api_key'),
        args.language,
        int(config.get('api', 'timeout'))
    )
    
    if not assessment_data:
        logger.error("Failed to get quality assessment data")
        sys.exit(1)
    
    # Save raw assessment data for reference
    raw_data_file = os.path.join(output_dir, f"quality_assessment_raw_{datetime.datetime.now().strftime('%Y%m%d')}.json")
    with open(raw_data_file, 'w') as f:
        json.dump(assessment_data, f, indent=2)
    
    # Load history file
    history_file = os.path.join(output_dir, config.get('monitoring', 'history_file'))
    history = load_quality_history(history_file)
    
    # Add current assessment to history
    history = add_assessment_to_history(history, assessment_data)
    
    # Save updated history
    save_quality_history(history_file, history)
    
    # Compare with previous if requested or if generating report
    comparison = {"is_first_assessment": True, "changes": {}}
    if args.compare or args.generate_report:
        comparison = compare_with_previous(history["assessments"][-1], history)
    
    # Generate charts if requested or if generating report
    charts = []
    if args.generate_report:
        charts = generate_charts(history, output_dir)
    
    # Generate HTML report if requested
    report_file = None
    if args.generate_report:
        report_file = generate_html_report(
            history["assessments"][-1],
            comparison,
            charts,
            output_dir,
            config
        )
    
    # Send notification if requested and report was generated
    if args.notify and report_file:
        send_notification(config, report_file, history["assessments"][-1], comparison)
    
    # Print summary to console
    current = history["assessments"][-1]
    print("\nDictionary Quality Assessment Summary:")
    print(f"Total entries: {current['total_entries']}")
    print(f"Quality score: {current['quality_score']}")
    print(f"Verified: {current['verified_percent']}%")
    print(f"Avg. completeness: {current['average_completeness']}")
    
    if not comparison["is_first_assessment"]:
        print("\nChanges since last assessment:")
        print(f"Quality score: {comparison['changes']['quality_score']['change']} ({comparison['changes']['quality_score']['percent_change']}%)")
        print(f"New entries: {comparison['changes']['total_entries']['change']}")
    
    if report_file:
        print(f"\nDetailed report saved to: {report_file}")
    
    sys.exit(0)

if __name__ == "__main__":
    main() 