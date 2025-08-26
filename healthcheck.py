#!/usr/bin/env python3
"""
Health check script for the Docker container.
This script verifies that the Flask app is running and responding to requests.
"""

import sys
import requests
import os

def health_check():
    """Perform a health check on the running Flask application."""
    port = os.getenv('PORT', '7860')
    host = 'localhost'
    url = f'http://{host}:{port}/'
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✓ Health check passed - Flask app is responding")
            return True
        else:
            print(f"✗ Health check failed - Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed - Connection error: {e}")
        return False

if __name__ == '__main__':
    success = health_check()
    sys.exit(0 if success else 1)