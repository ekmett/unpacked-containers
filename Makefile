all: build

build:
	cabal new-build

clean:
	rm -rf dist dist-newstyle

docs:
	cabal new-haddock

.PHONY:	all build clean docs
