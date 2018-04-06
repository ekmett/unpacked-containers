unpacked-containers
==

This package supplies a simple unpacked version of `Data.Set` and `Data.Map` using backpack.

This can remove a level of indirection on the heap and unpack your keys directly into nodes of your sets and maps.

The exported modules roughly follow the API of `containers 0.5.11`, but with all deprecated functions removed.

Note however, that all CPP has been removed relative to `containers`, because on one hand, use of backpack locks us to a current version of GHC,
and on the other there is a bug in GHC 8.2.2 that prevents the use of CPP in a module that uses backpack. This issue is resolved in GHC 8.4.1,
so as that comes into wider usage if we need to track `containers` API changes going forward and those need CPP we can just drop support for 8.2.2.

It is intended that you will remap the names of the modules. from `Set.*` or `Map.*` to some portion of the namespace that is peculiar to your
project, and so the module names are designed to be as short as possible, mirroring the usage of `containers` but with the `Data` prefix stripped off.

Usage
-----

To work this into an existing haskell project, you'll need to be on GHC >= 8.2.2, and use cabal >= 2. 

First build an internal library for your project that has a module that matches the `Key` signature.

```
module MyKey where

type Key = ()
```

You can put whatever you want in for `Key` as long as it is an instance of `Ord`.

Then in your cabal file you can set up your internal library as an extra named internal library (multiple library support was added in cabal 2).

```
library my-keys
  exposed-modules: MyKey
  build-depends: base
```

and in your library or executable that wants to work with sets or maps of that key type use


```
library
  build-depends: unpacked-containers, my-keys
  mixins: unpacked-containers (Set as MyKey.Set) requires (Key as MyKey)
```

If you need several `Set`s or `Map`s you can use several `mixins:` clauses.

If you need to expose the set type, remember you can use a `reexported-modules:` stanza.

Now you work with `MyKey.Set` as a monomorphic set type specific to the type of `Key` you specified earlier.

See the `executable unpacked-set-example` and `library example` sections in the `unpacked-containers.cabal` file for a minimal working example.

Documentation
==

To build haddocks for this project you need to run `cabal new-haddock` as `cabal-haddock` doesn't work.

Contact Information
-------------------

Contributions and bug reports are welcome!

Please feel free to contact me through github or on the #haskell IRC channel on irc.freenode.net.

-Edward Kmett
