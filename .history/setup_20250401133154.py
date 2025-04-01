from setuptools import setup, find_packages

setup(
    name="fil-relex",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "flask-cors",
        "flask-migrate",
        "flask-sqlalchemy",
        "psycopg2-binary",
        "python-dotenv",
    ],
) 