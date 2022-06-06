test:
	python ane_transformers/reference/test_transformer.py
	python ane_transformers/huggingface/test_distilbert.py

style:
	yapf -rip --verify ane_transformers