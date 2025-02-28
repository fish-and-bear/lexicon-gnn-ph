#!/bin/sh
set -e

# Simple health check that verifies the nginx service is running
curl -f http://localhost:80/health || exit 1 