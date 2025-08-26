#!/usr/bin/env python3
"""
Simple test script to verify the Flask application can start and respond to requests.
This is useful for Docker health checks and CI/CD validation.
"""

import sys
import subprocess
import time
import requests
import signal
import os

def test_app_startup():
    """Test that the Flask app can start and respond to requests."""
    print("Starting Flask app for testing...")
    
    # Set environment variables for testing
    env = os.environ.copy()
    env.update({
        'FLASK_ENV': 'development',
        'PORT': '5555',  # Use a different port for testing
        'HOST': '127.0.0.1'
    })
    
    # Start the Flask app in a subprocess
    try:
        process = subprocess.Popen(
            [sys.executable, 'app.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give the app time to start
        time.sleep(5)
        
        # Test if the app is responding
        try:
            response = requests.get('http://127.0.0.1:5555/', timeout=10)
            if response.status_code == 200:
                print("✓ Flask app started successfully and is responding")
                return True
            else:
                print(f"✗ Flask app responded with status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to connect to Flask app: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to start Flask app: {e}")
        return False
    finally:
        # Clean up the process
        if 'process' in locals():
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

if __name__ == '__main__':
    success = test_app_startup()
    sys.exit(0 if success else 1)