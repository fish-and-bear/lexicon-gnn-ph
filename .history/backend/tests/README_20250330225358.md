# Filipino Dictionary API Test Suite

This directory contains comprehensive test suites for the Filipino Dictionary API.

## Test Structure

The test suite consists of several components:

1. **Comprehensive Tests** (`test_api_comprehensive.py`)
   - Tests all API endpoints
   - Verifies response formats
   - Tests error handling
   - Checks CORS headers
   - Validates rate limiting

2. **Performance Tests** (`test_api_performance.py`)
   - Measures response times
   - Tests concurrent request handling
   - Monitors memory usage
   - Checks for memory leaks
   - Analyzes search performance

3. **Test Runner** (`run_tests.py`)
   - Executes all test suites
   - Generates detailed test reports
   - Tracks test execution time
   - Records test results

4. **Result Analysis** (`analyze_results.py`)
   - Analyzes test results
   - Generates performance metrics
   - Identifies areas for improvement
   - Creates detailed reports

5. **Test and Analysis Runner** (`test_and_analyze.py`)
   - Orchestrates the entire testing process
   - Runs tests and analysis in sequence
   - Checks for critical issues
   - Provides summary reports

## Setup

1. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

2. Set up test environment variables:
   ```bash
   export TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/fil_relex_test"
   export TEST_REDIS_URL="redis://localhost:6379/1"
   ```

## Running Tests

### Run All Tests and Analysis
```bash
python test_and_analyze.py
```

### Run Individual Test Suites
```bash
# Run comprehensive tests
python -m unittest test_api_comprehensive.py

# Run performance tests
python -m unittest test_api_performance.py
```

### Run Tests with Coverage Report
```bash
pytest --cov=../ --cov-report=html
```

## Test Reports

Test reports are generated in the following directories:

- `test_reports/`: Contains detailed test execution reports
- `analysis_reports/`: Contains analysis of test results
- `htmlcov/`: Contains HTML coverage reports

## Test Categories

### API Endpoints
- Word lookup
- Search functionality
- Baybayin processing
- Statistics
- Word relationships
- Etymology information

### Performance Metrics
- Response times
- Concurrent request handling
- Memory usage
- Database query performance
- Cache effectiveness

### Error Handling
- Invalid inputs
- Missing parameters
- Rate limiting
- Authentication errors
- Database errors

### Security
- API key validation
- CORS headers
- Rate limiting
- Input validation
- SQL injection prevention

## Continuous Integration

The test suite is designed to be run as part of a CI/CD pipeline. The `test_and_analyze.py` script will:

1. Run all tests
2. Generate detailed reports
3. Analyze results
4. Check for critical issues
5. Exit with appropriate status code

## Adding New Tests

When adding new tests:

1. Follow the existing test structure
2. Include both positive and negative test cases
3. Add performance tests for new endpoints
4. Update the analysis scripts if needed
5. Document new test cases in this README

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check TEST_DATABASE_URL environment variable
   - Verify database is running
   - Check database credentials

2. **Redis Connection Errors**
   - Check TEST_REDIS_URL environment variable
   - Verify Redis is running
   - Check Redis configuration

3. **Test Timeouts**
   - Adjust timeout settings in test files
   - Check system resources
   - Monitor database performance

4. **Memory Issues**
   - Check for memory leaks in tests
   - Monitor system resources
   - Adjust test batch sizes

### Getting Help

For issues with the test suite:

1. Check the test reports for detailed error information
2. Review the analysis reports for performance issues
3. Check the logs for runtime errors
4. Contact the development team for support

## Contributing

When contributing to the test suite:

1. Follow the existing code style
2. Add appropriate documentation
3. Include test cases for new features
4. Update the README as needed
5. Ensure all tests pass before submitting

## License

This test suite is part of the Filipino Dictionary project and follows the same licensing terms. 