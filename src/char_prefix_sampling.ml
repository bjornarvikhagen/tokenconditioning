module Error = struct
  type t =
    | EmptyPrefix
    | MaxAttemptsExceeded of { attempts: int }
    | InvalidPrefix of { generated: string; prefix: string }
    | TokenizerError of string

  let to_string = function
    | EmptyPrefix -> "prefix cannot be empty"
    | MaxAttemptsExceeded { attempts } ->
        Printf.sprintf "failed to find valid token after %d attempts" attempts
    | InvalidPrefix { generated; prefix } ->
        Printf.sprintf "generated text '%s' doesn't match prefix '%s'" 
          generated prefix
    | TokenizerError msg -> msg
end

type token = {
  value: string;
  logprob: float;
  metadata: (string * string) list;  (* extensible metadata *)
}

module type TokenSampler = sig
  type error
  val sample_next_token : token list -> (token, error) result
  val get_error_message : error -> string
end

module Make (S : TokenSampler) = struct
  type sampler = {
    max_attempts: int;
    temperature: float;
    top_p: float option;
    top_k: int option;
  }

  let create 
    ?(max_attempts=1000) 
    ?(temperature=1.0)
    ?top_p
    ?top_k
    () = {
      max_attempts;
      temperature;
      top_p;
      top_k;
    }

  let get_remaining_prefix prefix generated_tokens =
    if String.length prefix = 0 then 
      Error Error.EmptyPrefix
    else
      let generated_text = 
        String.concat "" (List.map (fun t -> t.value) generated_tokens) in
      if String.starts_with ~prefix:prefix generated_text then 
        Ok None
      else if String.starts_with ~prefix:generated_text prefix then
        Ok (Some (String.sub prefix 
          (String.length generated_text)
          (String.length prefix - String.length generated_text)))
      else
        Error (Error.InvalidPrefix { 
          generated = generated_text; 
          prefix = prefix 
        })

  let is_valid_next_token token = function
    | None -> true
    | Some remaining_prefix ->
        String.starts_with ~prefix:remaining_prefix token.value ||
        String.starts_with ~prefix:token.value remaining_prefix

  let rec attempt_sample ~sampler remaining_prefix attempts context =
    if attempts = 0 then
      Error (Error.MaxAttemptsExceeded { attempts = sampler.max_attempts })
    else
      match S.sample_next_token context with
      | Error e -> Error (Error.TokenizerError (S.get_error_message e))
      | Ok candidate ->
          let candidate_text = candidate.value in
          match remaining_prefix with
          | None -> Ok candidate
          | Some rp ->
              if List.length context = 0 then
                (* For first token, must start with or be started by the prefix *)
                if String.starts_with ~prefix:rp candidate_text ||
                   String.starts_with ~prefix:candidate_text rp then
                  Ok candidate
                else
                  attempt_sample ~sampler remaining_prefix (attempts - 1) context
              else
                (* For subsequent tokens, must continue the prefix *)
                if String.starts_with ~prefix:rp candidate_text then
                  Ok candidate
                else
                  attempt_sample ~sampler remaining_prefix (attempts - 1) context

  let sample_sequence 
    sampler 
    ?(max_tokens=100) 
    ?stop_tokens 
    ?(min_tokens=1)
    prefix =
    let stop_tokens = Option.value stop_tokens ~default:[] in
    let rec generate acc count =
      if count >= max_tokens then 
        Ok (List.rev acc)
      else if count >= min_tokens && 
              List.exists (fun t -> List.mem t.value stop_tokens) acc then
        Ok (List.rev acc)
      else
        match get_remaining_prefix prefix acc with
        | Error e -> Error e
        | Ok remaining_prefix ->
            match attempt_sample ~sampler remaining_prefix sampler.max_attempts acc with
            | Error e -> Error e
            | Ok token ->
                if remaining_prefix = None && 
                   count >= min_tokens && 
                   String.trim token.value = "" then
                  Ok (List.rev (token :: acc))
                else
                  generate (token :: acc) (count + 1)
    in
    generate [] 0

  (* Utility functions *)
  let get_text tokens =
    String.concat "" (List.map (fun t -> t.value) tokens)

  let get_logprob tokens =
    List.fold_left (fun acc t -> acc +. t.logprob) 0. tokens

  let get_token_count = List.length

  let get_metadata tokens key =
    List.filter_map (fun t -> 
      List.assoc_opt key t.metadata
    ) tokens
end 