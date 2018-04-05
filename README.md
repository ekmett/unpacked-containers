unpacked-containers
==

This package supplies a simple unpacked version of Data.Set and Data.Map using backpack.

This can remove a level of indirection on the heap and unpack your keys directly into nodes of your sets and maps.

See `unpacked-set-example` for a tiny example of usage.

Documentation
==

To build haddocks for this project you need to configure with

```
cabal configure -fhaddock
```

and then you can `cabal new-haddock` or `cabal haddock` as usual.

This will produce viable documentation, but in that form the library is broken.

Contact Information
-------------------

Contributions and bug reports are welcome!

Please feel free to contact me through github or on the #haskell IRC channel on irc.freenode.net.

-Edward Kmett
