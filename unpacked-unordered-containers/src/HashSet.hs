{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE Trustworthy #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}

------------------------------------------------------------------------
-- |
-- Module      :  HashSet
-- Copyright   :  2011 Bryan O'Sullivan
-- License     :  BSD-style
-- Maintainer  :  johan.tibell@gmail.com
-- Stability   :  provisional
-- Portability :  portable
--
-- A set of /hashable/ values.  A set cannot contain duplicate items.
-- A 'HashSet' makes no guarantees as to the order of its elements.
--
-- The implementation is based on /hash array mapped trie/.  A
-- 'HashSet' is often faster than other tree-based set types,
-- especially when value comparison is expensive, as in the case of
-- strings.
--
-- Many operations have a average-case complexity of /O(log n)/.  The
-- implementation uses a large base (i.e. 16) so in practice these
-- operations are constant time.

module HashSet
    (
      HashSet

    -- * Construction
    , empty
    , singleton

    -- * Combine
    , union
    , unions

    -- * Basic interface
    , null
    , size
    , member
    , insert
    , delete

    -- * Transformations
    , map

      -- * Difference and intersection
    , difference
    , intersection

    -- * Folds
    , foldl'
    , foldr

    -- * Filter
    , filter

    -- * Conversions

    -- ** Lists
    , toList
    , fromList

    -- * HashMaps
    , toMap
    , fromMap
    ) where

import Control.DeepSeq (NFData(..))
import Data.Data hiding (Typeable)
import Data.Hashable (Hashable(hashWithSalt))
import Data.Semigroup (Semigroup(..))
import GHC.Exts (build)
import Prelude hiding (filter, foldr, map, null)
import qualified Data.List as List
import Text.Read
import qualified GHC.Exts as Exts

import qualified HashMap.Lazy as H
import HashMap.Base (HashMap, foldrWithKey, equalKeys)

import Key

-- | A set of values.  A set cannot contain duplicate values.
newtype HashSet = HashSet { asMap :: HashMap () }

instance NFData Key => NFData HashSet where
    rnf = rnf . asMap
    {-# INLINE rnf #-}

instance Eq HashSet where
    HashSet a == HashSet b = equalKeys (==) a b
    {-# INLINE (==) #-}

instance Ord Key => Ord HashSet where
    compare (HashSet a) (HashSet b) = compare a b
    {-# INLINE compare #-}

instance Semigroup HashSet where
    (<>) = union
    {-# INLINE (<>) #-}

instance Monoid HashSet where
    mempty = empty
    {-# INLINE mempty #-}
    mappend = (<>)
    {-# INLINE mappend #-}

instance Read Key => Read HashSet where
    readPrec = parens $ prec 10 $ do
      Ident "fromList" <- lexP
      xs <- readPrec
      return (fromList xs)

    readListPrec = readListPrecDefault

instance Show Key => Show HashSet where
    showsPrec d m = showParen (d > 10) $
      showString "fromList " . shows (toList m)

instance Data Key => Data HashSet where
    gfoldl f z m   = z fromList `f` toList m
    toConstr _     = fromListConstr
    gunfold k z c  = case constrIndex c of
        1 -> k (z fromList)
        _ -> error "gunfold"
    dataTypeOf _   = hashSetDataType

instance Hashable HashSet where
    hashWithSalt salt = hashWithSalt salt . asMap

fromListConstr :: Constr
fromListConstr = mkConstr hashSetDataType "fromList" [] Prefix

hashSetDataType :: DataType
hashSetDataType = mkDataType "HashSet" [fromListConstr]

-- | /O(1)/ Construct an empty set.
empty :: HashSet
empty = HashSet H.empty

-- | /O(1)/ Construct a set with a single element.
singleton :: Key -> HashSet
singleton a = HashSet (H.singleton a ())
{-# INLINEABLE singleton #-}

-- | /O(1)/ Convert to the equivalent 'HashMap'.
toMap :: HashSet -> HashMap ()
toMap = asMap

-- | /O(1)/ Convert from the equivalent 'HashMap'.
fromMap :: HashMap () -> HashSet
fromMap = HashSet

-- | /O(n+m)/ Construct a set containing all elements from both sets.
--
-- To obtain good performance, the smaller set must be presented as
-- the first argument.
union :: HashSet -> HashSet -> HashSet
union s1 s2 = HashSet $ H.union (asMap s1) (asMap s2)
{-# INLINE union #-}

-- TODO: Figure out the time complexity of 'unions'.

-- | Construct a set containing all elements from a list of sets.
unions :: [HashSet] -> HashSet
unions = List.foldl' union empty
{-# INLINE unions #-}

-- | /O(1)/ Return 'True' if this set is empty, 'False' otherwise.
null :: HashSet -> Bool
null = H.null . asMap
{-# INLINE null #-}

-- | /O(n)/ Return the number of elements in this set.
size :: HashSet -> Int
size = H.size . asMap
{-# INLINE size #-}

-- | /O(log n)/ Return 'True' if the given value is present in this
-- set, 'False' otherwise.
member :: Key -> HashSet -> Bool
member a s = case H.lookup a (asMap s) of
               Just _ -> True
               _      -> False
{-# INLINABLE member #-}

-- | /O(log n)/ Add the specified value to this set.
insert :: Key -> HashSet -> HashSet
insert a = HashSet . H.insert a () . asMap
{-# INLINABLE insert #-}

-- | /O(log n)/ Remove the specified value from this set if
-- present.
delete :: Key -> HashSet -> HashSet
delete a = HashSet . H.delete a . asMap
{-# INLINABLE delete #-}

-- | /O(n)/ Transform this set by applying a function to every value.
-- The resulting set may be smaller than the source.
map :: (Key -> Key) -> HashSet -> HashSet
map f = fromList . List.map f . toList
{-# INLINE map #-}

-- | /O(n)/ Difference of two sets. Return elements of the first set
-- not existing in the second.
difference :: HashSet -> HashSet -> HashSet
difference (HashSet a) (HashSet b) = HashSet (H.difference a b)
{-# INLINABLE difference #-}

-- | /O(n)/ Intersection of two sets. Return elements present in both
-- the first set and the second.
intersection :: HashSet -> HashSet -> HashSet
intersection (HashSet a) (HashSet b) = HashSet (H.intersection a b)
{-# INLINABLE intersection #-}

-- | /O(n)/ Reduce this set by applying a binary operator to all
-- elements, using the given starting value (typically the
-- left-identity of the operator).  Each application of the operator
-- is evaluated before before using the result in the next
-- application.  This function is strict in the starting value.
foldl' :: (a -> Key -> a) -> a -> HashSet -> a
foldl' f z0 = H.foldlWithKey' g z0 . asMap
  where g z k _ = f z k
{-# INLINE foldl' #-}

-- | /O(n)/ Reduce this set by applying a binary operator to all
-- elements, using the given starting value (typically the
-- right-identity of the operator).
foldr :: (Key -> a -> a) -> a -> HashSet -> a
foldr f z0 = foldrWithKey g z0 . asMap
  where g k _ z = f k z
{-# INLINE foldr #-}

-- | /O(n)/ Filter this set by retaining only elements satisfying a
-- predicate.
filter :: (Key -> Bool) -> HashSet -> HashSet
filter p = HashSet . H.filterWithKey q . asMap
  where q k _ = p k
{-# INLINE filter #-}

-- | /O(n)/ Return a list of this set's elements.  The list is
-- produced lazily.
toList :: HashSet -> [Key]
toList t = build (\ c z -> foldrWithKey ((const .) c) z (asMap t))
{-# INLINE toList #-}

-- | /O(n*min(W, n))/ Construct a set from a list of elements.
fromList :: [Key] -> HashSet
fromList = HashSet . List.foldl' (\ m k -> H.insert k () m) H.empty
{-# INLINE fromList #-}

instance Exts.IsList HashSet where
    type Item HashSet = Key
    fromList = fromList
    toList   = toList
