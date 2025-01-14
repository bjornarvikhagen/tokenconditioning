# character prefix conditional sampling

an efficient implementation of character-prefix-conditional sampling for autoregressive language models in ocaml.

## the problem

when using a language model for code completion, we typically want the model to produce a completion that begins with what the user has typed. however, modern language models operate on sequences of tokens, not characters, creating a mismatch when the user's cursor isn't at a token boundary.

instead, we need an algorithm that samples a sequence of tokens conditional on a prefix of characters, rather than the more typical case of sampling conditional on a prefix of tokens.

### mathematical formulation

we want to sample a sequence of tokens s = t₁,t₂,…,tₙ from a distribution specified by an autoregressive model p(s) given by:

```
p(s) = p(t₁,t₂,…,tₙ) = ∏ₖ₌₁ⁿ p(tₖ|t₁,…,tₖ₋₁)
```

subject to the constraint that s starts with a character prefix P, i.e. P is a prefix of repr(t₁) + repr(t₂) + ⋯ + repr(tₙ), where + means string concatenation and repr maps a token to the characters it represents.

we define q(s) = p(s|s starts with P). it's sufficient to find a way to sample autoregressively from q(s), that is, to sample from q(tₖ|t₁,…,tₖ₋₁) for each k.

## solution

this implementation provides an efficient algorithm for sampling tokens conditional on a character prefix by using ocaml's strong type system and module system. features:

- minimal calls to the underlying language model through rejection sampling
- proper handling of partial token matches
- configurable stopping conditions and sampling parameters
- robust error handling with result types
- comprehensive test suite using alcotest
- functorial design for dependency injection

## usage

first, implement the `TokenSampler` interface for your model:

```ocaml
module GPT2Sampler : TokenSampler = struct
  type error = 
    | ModelError of string
    | InvalidState of string

  let get_error_message = function
    | ModelError msg -> msg
    | InvalidState msg -> msg

  let sample_next_token context =
    try
      let logits = model.forward context in
      let logits = 
        match temperature with
        | Some t -> scale_logits logits t
        | None -> logits in
      let logits = 
        match top_p with
        | Some p -> nucleus_sampling logits p
        | None -> logits in
      let token = sample_token logits in
      Ok { 
        value = token.text;
        logprob = token.logprob;
        metadata = [
          "pos", string_of_int (List.length context);
          "token_id", string_of_int token.id
        ]
      }
    with
    | Model_error msg -> Error (ModelError msg)
    | Invalid_state msg -> Error (InvalidState msg)
end
```

then create and use the prefix sampler:

```ocaml
(* create sampler with your settings *)
module PrefixSampler = Make(GPT2Sampler)
let sampler = PrefixSampler.create 
  ~max_attempts:100    (* max tries per token *)
  ~temperature:0.8     (* creativity vs. consistency *)
  ~top_p:(Some 0.9)    (* nucleus sampling threshold *)
  ~top_k:(Some 50)     (* top-k sampling limit *)
  ()

(* sample completion for "def my_func" *)
match PrefixSampler.sample_sequence 
  sampler
  ~max_tokens:50       (* max length *)
  ~min_tokens:10       (* min length *)
  ~stop_tokens:["\n"; ":"] (* stop at newline or colon *)
  "def my_func" with
| Ok tokens -> 
    let text = PrefixSampler.get_text tokens in
    Printf.printf "completion: %s\n" text;
    let logprob = PrefixSampler.get_logprob tokens in
    Printf.printf "total logprob: %.3f\n" logprob;
    (* get position metadata *)
    let positions = PrefixSampler.get_metadata tokens "pos" in
    List.iteri (fun i pos -> 
      Printf.printf "token %d: pos %s\n" i pos
    ) positions
| Error e -> 
    Printf.eprintf "error: %s\n" (Error.to_string e)
```

## implementation details

the algorithm uses rejection sampling with early stopping optimizations:

1. for the first token:
   - accepts if token exactly matches prefix
   - accepts if token is a prefix of the target
   - accepts if target is a prefix of token

2. for subsequent tokens:
   - if prefix is satisfied, samples normally
   - if prefix remains, requires token to start with remaining prefix

this approach maintains proper probability distributions while being computationally efficient.

## building and testing

requires ocaml ≥ 4.14.0 and opam. to build:

```bash
# install dependencies
opam install . --deps-only --with-test

# build
dune build

# run tests
dune runtest
```

## module structure

- `TokenSampler`: abstract interface for token sampling
- `Make`: functor that takes a `TokenSampler` and produces a prefix-conditional sampler
- `Error`: custom error types for better error handling
- utility functions for text, logprob, and metadata extraction

## features

- proper error handling with custom error types
- extensible token metadata support
- configurable sampling parameters (temperature, top_p, top_k)
- min/max token controls
- stop token support
- utility functions for common operations

## credits

this problem was originally posed by jacob at cursor as part of a series exploring code completion algorithms. 