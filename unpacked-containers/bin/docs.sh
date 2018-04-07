#!/bin/sh
cabal new-haddock --haddock-for-hackage --haddock-option=--hyperlinked-source
cabal upload -d dist-newstyle/unpacked-containers-*-docs.tar.gz --publish
