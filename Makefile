all: build

build:
	cabal configure
	cabal new-build

clean:
	rm -rf dist dist-newstyle

docs:
	cabal configure -fhaddock
	cabal new-haddock

.PHONY:	all build clean docs
