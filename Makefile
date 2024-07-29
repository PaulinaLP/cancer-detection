setup:
	pipenv install --dev
	pre-commit install

quality_checks:
	isort .
	black .
	pylint --recursive=y .

unit_tests:
	cd webservice && pytest tests/test_preprocess.py

integration_tests:
	cd webservice && pytest tests/test_integration.py

build: quality_checks unit_tests integration_tests
	docker-compose -f docker-compose-webservice.yaml up --build -d



