mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
copy_data = (mkdir -p $(mkfile_dir)nnsplit/data && cp -a $(mkfile_dir)../data/* $(mkfile_dir)nnsplit/data/)

test:
	$(call copy_data)
	poetry run pytest -s tests

build:
	# Workaround for https://github.com/python-poetry/poetry/issues/1338#issuecomment-571618450
	mv .gitignore .gitignore-invalid
	poetry build || mv .gitignore-invalid .gitignore
	mv .gitignore-invalid .gitignore

publish:
	$(call copy_data)
	make build
	poetry publish
