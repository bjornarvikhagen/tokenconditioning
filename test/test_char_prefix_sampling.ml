[@@@ocaml.warning "-32"]
open Alcotest
open Char_prefix_sampling

module Sampler = Make(Mock_sampler.MockSampler)
let sampler = Sampler.create 
  ~max_attempts:1000 
  ~temperature:0.8
  ~top_p:0.9
  ~top_k:50
  ()

let setup_vocabulary () =
  Mock_sampler.MockSampler.set_vocabulary [
    ("def", -1.0);  (* Put matching tokens first *)
    ("defun", -1.0);
    ("define", -1.0);
    ("class", -1.0);
    ("foo", -1.0);
    ("bar", -1.0);
    ("(", -1.0);
    (")", -1.0);
    (":", -1.0);
    (" ", -1.0);
  ]

(* Custom test helpers *)
let check_error ~expected = function
  | Ok _ -> fail "Expected error but got Ok"
  | Error e -> check string "error message" expected (Error.to_string e)

let check_prefix_match prefix = function
  | Error e -> fail (Error.to_string e)
  | Ok tokens ->
      let text = Sampler.get_text tokens in
      check bool "starts with prefix" true (String.starts_with ~prefix text)

let test_empty_prefix () =
  check_error 
    ~expected:"prefix cannot be empty"
    (Sampler.sample_sequence sampler "")

let test_basic_prefix_satisfaction () =
  setup_vocabulary ();
  let prefix = "def" in
  let result = Sampler.sample_sequence sampler prefix in
  check_prefix_match prefix result;
  match result with
  | Error e -> fail (Error.to_string e)
  | Ok tokens ->
      (* Check metadata *)
      let positions = Sampler.get_metadata tokens "position" in
      check bool "has positions" true (List.length positions > 0);
      let first_tokens = Sampler.get_metadata tokens "is_first" in
      check string "first token marked" "true" (List.hd first_tokens)

let test_partial_token_prefix () =
  setup_vocabulary ();
  let prefix = "de" in
  check_prefix_match prefix (Sampler.sample_sequence sampler prefix)

let test_stop_tokens () =
  setup_vocabulary ();
  let stop_tokens = [":"] in
  match Sampler.sample_sequence sampler ~stop_tokens "def" with
  | Error e -> fail (Error.to_string e)
  | Ok tokens ->
      let text = Sampler.get_text tokens in
      check bool "starts with def" true (String.starts_with ~prefix:"def" text);
      (* Check logprob *)
      let total_logprob = Sampler.get_logprob tokens in
      check bool "has valid logprob" true (total_logprob < 0.)

let test_max_tokens_limit () =
  setup_vocabulary ();
  let max_tokens = 3 in
  match Sampler.sample_sequence sampler ~max_tokens "def" with
  | Error e -> fail (Error.to_string e)
  | Ok tokens ->
      check int "respects max tokens" max_tokens (Sampler.get_token_count tokens)

let test_min_tokens () =
  setup_vocabulary ();
  let min_tokens = 5 in
  match Sampler.sample_sequence sampler ~min_tokens "def" with
  | Error e -> fail (Error.to_string e)
  | Ok tokens ->
      check bool "respects min tokens" true 
        (Sampler.get_token_count tokens >= min_tokens)

let test_invalid_prefix () =
  Mock_sampler.MockSampler.set_vocabulary [("xyz", -1.0)];
  match Sampler.sample_sequence sampler "abc" with
  | Ok _ -> fail "Expected error for impossible prefix"
  | Error e -> 
      match e with
      | Error.MaxAttemptsExceeded _ -> ()
      | _ -> fail "Expected MaxAttemptsExceeded error"

let () =
  Random.self_init ();
  run "Character Prefix Sampling" [
    "basic", [
      test_case "empty prefix raises" `Quick test_empty_prefix;
      test_case "basic prefix satisfaction" `Quick test_basic_prefix_satisfaction;
      test_case "partial token prefix" `Quick test_partial_token_prefix;
      test_case "stop tokens" `Quick test_stop_tokens;
      test_case "max tokens limit" `Quick test_max_tokens_limit;
      test_case "min tokens requirement" `Quick test_min_tokens;
      test_case "invalid prefix" `Quick test_invalid_prefix;
    ];
  ] 