[@@@ocaml.warning "-32-34"]
open Char_prefix_sampling

module MockSampler : sig
  include TokenSampler
  val set_vocabulary : (string * float) list -> unit
end = struct
  type error = string
  let get_error_message e = e

  type vocabulary = (string * float) list

  let vocabulary = ref [
    ("def", -1.0);
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

  let set_vocabulary v = vocabulary := v

  let find_matching_token prefix tokens =
    List.find_opt (fun (value, _) -> 
      String.starts_with ~prefix value || 
      String.starts_with ~prefix:value prefix
    ) tokens

  let sample_next_token context =
    let tokens = !vocabulary in
    match context with
    | [] -> 
        (* For first token, try to find one that matches the prefix *)
        (match List.hd tokens with
        | (value, logprob) -> 
            Ok { 
              value; 
              logprob; 
              metadata = [
                "position", string_of_int (List.length context);
                "is_first", "true"
              ] 
            })
    | _ -> 
        (* For subsequent tokens, just take the first one *)
        let (value, logprob) = List.hd tokens in
        Ok { 
          value; 
          logprob; 
          metadata = [
            "position", string_of_int (List.length context);
            "is_first", "false"
          ] 
        }
end 