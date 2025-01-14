(lang dune 3.0)
(name char_prefix_sampling)
(version 0.1.0)

(generate_opam_files true)

(package
 (name char_prefix_sampling)
 (synopsis "Character prefix conditional sampling for language models")
 (description "An efficient implementation of character-prefix-conditional sampling for autoregressive language models")
 (depends
  (ocaml (>= 4.14.0))
  (dune (>= 3.0))
  (alcotest :with-test))) 