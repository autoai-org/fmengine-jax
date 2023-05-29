format:
	autoflake -i **/*.py
	yapf -i **/*.py

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
