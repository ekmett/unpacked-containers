module Main where

import Int.Set

ten :: Set
ten = fromList [1..10]

main :: IO ()
main = print ten
