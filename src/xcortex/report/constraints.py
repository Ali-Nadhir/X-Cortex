"""Constrain local report syntax; clinical grounding is verified separately."""
from __future__ import annotations

import threading
import weakref

from ..core.execution import check_cancelled
from .schema import REPORT_SCHEMA


# The tokenizer bridge below adapts LM Format Enforcer's Transformers integration
# without importing its Transformers-4-only module in release 0.11.3.
# https://github.com/noamgat/lm-format-enforcer/blob/main/lmformatenforcer/integrations/transformers.py
#
# MIT License
# Copyright (c) 2023 Noam Gat
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

_TOKENIZER_DATA = weakref.WeakKeyDictionary()
_TOKENIZER_DATA_LOCK = threading.Lock()


def _decode(tokenizer, token_ids):
    # Disabling cleanup is necessary for exact JSON syntax and whitespace.
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)


def _build_tokenizer_data(tokenizer):
    from lmformatenforcer import TokenEnforcerTokenizerData

    vocabulary = tokenizer.get_vocab()
    token_ids = sorted(set(vocabulary.values()))
    if not token_ids:
        raise ValueError("Structured report tokenizer has an empty vocabulary")
    special_ids = set(tokenizer.all_special_ids)
    # Transformers 5's Gemma tokenizer does not include every registered special
    # AddedToken (including end_of_turn) in all_special_ids.
    special_ids.update(
        token_id for token_id, token in tokenizer.added_tokens_decoder.items()
        if getattr(token, "special", False)
    )
    eos = tokenizer.eos_token_id
    eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    eos_ids = [token_id for token_id in eos_ids if isinstance(token_id, int)]
    # Gemma's generation config also stops at its registered end-of-turn token.
    # It must be available only after the JSON parser accepts a complete report.
    end_of_turn = vocabulary.get("<end_of_turn>")
    if end_of_turn in special_ids and end_of_turn not in eos_ids:
        eos_ids.append(end_of_turn)
    if not eos_ids:
        raise ValueError("Structured report tokenizer requires an EOS token")
    special_ids.update(eos_ids)

    anchor = tokenizer.encode("0", add_special_tokens=False)
    anchor_text = _decode(tokenizer, anchor)
    if not anchor or anchor_text != "0":
        raise ValueError("Structured report tokenizer cannot round-trip the decoding anchor")
    regular_tokens = []
    for index, token_id in enumerate(token_ids):
        if index % 256 == 0:
            check_cancelled()
        if token_id in special_ids:
            continue
        # SentencePiece drops a word-initial space when decoding a token alone;
        # decoding after an anchor preserves its actual continuation spelling.
        continuation = _decode(tokenizer, [*anchor, token_id])[len(anchor_text):]
        standalone = _decode(tokenizer, [token_id])
        regular_tokens.append((token_id, continuation, len(continuation) > len(standalone)))

    tokenizer_ref = weakref.ref(tokenizer)

    def decode_tokens(ids):
        current = tokenizer_ref()
        if current is None:
            raise RuntimeError("The structured report tokenizer is no longer available")
        # Byte-fallback tokenizers may finish a chunk with an incomplete UTF-8
        # sequence. Defer its replacement character until enough bytes arrive.
        return _decode(current, ids).rstrip("\ufffd")

    return TokenEnforcerTokenizerData(
        regular_tokens, decode_tokens, eos_ids, use_bitmask=False, vocab_size=max(token_ids) + 1
    )


def _tokenizer_data(tokenizer):
    # The tree for Gemma's 262k vocabulary is costly; share it while retaining a
    # fresh parser/enforcer for every request. The decoder holds only a weakref.
    with _TOKENIZER_DATA_LOCK:
        data = _TOKENIZER_DATA.get(tokenizer)
        if data is None:
            data = _build_tokenizer_data(tokenizer)
            _TOKENIZER_DATA[tokenizer] = data
        return data


class _ReportPrefixAllowedTokens:
    def __init__(self, token_enforcer):
        self.token_enforcer = token_enforcer

    def __call__(self, batch_id, input_ids):
        check_cancelled()
        return self.token_enforcer.get_allowed_tokens(input_ids.tolist()).allowed_tokens

    def is_complete(self, input_ids) -> bool:
        """Whether this generation has closed its entire structured JSON value.

        LMFE substitutes ForceStopParser after internal failures. That parser
        permits EOS but does not establish a complete JSON report, so it must
        never trigger our structured-completion stopping criterion.
        """
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.characterlevelparser import UnionParser

        check_cancelled()
        sequence = input_ids.tolist()
        self.token_enforcer.get_allowed_tokens(sequence)
        state = self.token_enforcer.prefix_states.get(tuple(sequence))
        if state is None:
            return False

        def accepted_json(parser) -> bool:
            if isinstance(parser, JsonSchemaParser):
                return parser.can_end()
            if isinstance(parser, UnionParser):
                return any(accepted_json(branch) for branch in parser.parsers)
            return False

        return accepted_json(state.parser)


def build_report_constraint(tokenizer, response_schema=None):
    """Allow only continuations of the actual ClinicalReport JSON schema.

    This is an ordinary Transformers decoding constraint, not a model-specific
    thinking switch. Free-text chat uses the backend's final-answer filter.
    A fresh parser is required for every generation/request.
    """
    try:
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser, TokenEnforcer
    except ImportError as exc:
        raise RuntimeError(
            "Structured MedGemma reports require lm-format-enforcer; install the project's models extra"
        ) from exc
    schema = REPORT_SCHEMA if response_schema is None else response_schema
    required = schema.get("required", [])
    choice = schema.get("properties", {}).get("answer_choice", {})
    ordered_mcq = bool(required and required[0] == "answer_choice"
                       and choice.get("type") == "string" and choice.get("enum"))
    parser = JsonSchemaParser(schema)
    enforcer = TokenEnforcer(_tokenizer_data(tokenizer), parser)
    # LMFE 0.11.3 overwrites parser.config in TokenEnforcer.__init__, discarding
    # constructor-supplied field-order/whitespace limits. Apply the request
    # settings afterwards while preserving its tokenizer-derived alphabet.
    parser.config = CharacterLevelParserConfig(
        alphabet=parser.config.alphabet,
        max_consecutive_whitespaces=2 if ordered_mcq else parser.config.max_consecutive_whitespaces,
        force_json_field_order=ordered_mcq,
    )
    parser.context.alphabet_without_quotes = parser.config.alphabet.replace('"', '')
    return _ReportPrefixAllowedTokens(enforcer)
