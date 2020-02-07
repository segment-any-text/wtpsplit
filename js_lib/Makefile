.PHONY: test

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
copy_data = (mkdir -p $(mkfile_dir)example/nnsplit/data && cp -a $(mkfile_dir)../data/* $(mkfile_dir)example/nnsplit/data/)

install:
	$(call copy_data)
	npm install

test:
	$(call copy_data)
	./node_modules/mocha/bin/mocha

develop:
	$(call copy_data)
	npx webpack-dev-server

build:
	$(call copy_data)
	npx webpack