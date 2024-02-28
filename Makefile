format:
	isort *.py && black --line-length=79 *.py
	
jupytext:
	make format
	jupytext --set-formats ipynb,py:percent main.py

run:
	make format
	python main.py