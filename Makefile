format:
	autoflake -i **/*.py
	yapf -i **/*.py

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

build:
	python -m build --wheel

publish-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish:
	twine upload dist/*

build-sif:
	@sudo singularity build build/fmtrainer.sif build/build.def