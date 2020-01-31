mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
copy_data = (mkdir -p $(mkfile_dir)nnsplit/data && cp -a $(mkfile_dir)../data/* $(mkfile_dir)nnsplit/data/)

test:
	$(call copy_data)
	poetry run pytest -s tests

publish:
	$(call copy_data)
	poetry publish
