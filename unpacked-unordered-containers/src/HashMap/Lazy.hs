{-# LANGUAGE Trustworthy #-}
------------------------------------------------------------------------
-- |
-- Module      :  HashMap.Lazy
-- Copyright   :  (C) 2010-2012 Johan Tibell
--                (C) 2018 Edward Kmett
-- License     :  BSD-style
-- Maintainer  :  ekmett@gmail.com
-- Stability   :  experimental
-- Portability :  nonportable
--
-- A map from /hashable/ keys to values.  A map cannot contain
-- duplicate keys; each key can map to at most one value.  A 'HashMap'
-- makes no guarantees as to the order of its elements.
--
-- The implementation is based on /hash array mapped tries/.  A
-- 'HashMap' is often faster than other tree-based set types,
-- especially when key comparison is expensive, as in the case of
-- strings.
--
-- Many operations have a average-case complexity of /O(log n)/.  The
-- implementation uses a large base (i.e. 16) so in practice these
-- operations are constant time.
module HashMap.Lazy
    (
      -- * Strictness properties
      -- $strictness

      HashMap

      -- * Construction
    , empty
    , singleton

      -- * Basic interface
    , HM.null
    , size
    , member
    , HM.lookup
    , lookupDefault
    , (!)
    , insert
    , insertWith
    , delete
    , adjust
    , update
    , alter

      -- * Combine
      -- ** Union
    , union
    , unionWith
    , unionWithKey
    , unions

      -- * Transformations
    , HM.map
    , mapWithKey
    , traverseWithKey

      -- * Difference and intersection
    , difference
    , differenceWith
    , intersection
    , intersectionWith
    , intersectionWithKey

      -- * Folds
    , foldl'
    , foldlWithKey'
    , HM.foldr
    , foldrWithKey

      -- * Filter
    , HM.filter
    , filterWithKey
    , mapMaybe
    , mapMaybeWithKey

      -- * Conversions
    , keys
    , elems

      -- ** Lists
    , toList
    , fromList
    , fromListWith
    ) where

import HashMap.Base as HM

-- $strictness
--
-- This module satisfies the following strictness property:
--
-- * Key arguments are evaluated to WHNF
