setup:
	pipenv install --dev
	pre-commit install

quality_checks:
	isort .
	black .
	pylint --recursive=y .

unit_tests:
	pytest webservice/tests/test_preprocess.py

integration_tests:
	pytest webservice/tests/test_integration.py

build: quality_checks unit_tests integration_tests
	docker-compose -f docker-compose-webservice.yml up --build -d