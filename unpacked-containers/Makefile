all: build

build:
	cabal new-build

clean:
	rm -rf dist dist-newstyle

docs:
	cabal new-haddock
	cp -aRv dist-newstyle/build/*/*/unpacked-containers-0/doc/html/unpacked-containers/* docs
	cd docs && git commit -a -m "update haddocks" && git push && cd ..

.PHONY:	all build clean docs
