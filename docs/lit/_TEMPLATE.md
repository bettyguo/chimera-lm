# [Author year] — [Title]

> **Status:** TODO — unread. Copy this template to `{slug}.md` and fill in
> while reading. Do not commit a stub with unfilled sections as if it were
> a real note.

## Primitive operation

[The one thing this paper defines. E.g.: "selective scan with input-dependent
A, B, C matrices over a state-space recurrence."]

## Complexity

- Prefill / train: [e.g. O(T·d·S) with parallel scan]
- Decode per token: [e.g. O(d·S)]
- KV / state memory at step T: [e.g. O(L·d·S) constant in T]

## Failure mode the paper itself admits

[Quote or paraphrase the limitations section. Don't invent failure modes;
report what the authors said. E.g.: "associative recall degrades vs.
attention as T grows; copy task accuracy drops above T=256."]

## Empirically strongest result

[Headline benchmark + number + scale. E.g.: "3B Mamba matches 3B Transformer
on Pile validation perplexity at 1T tokens; 5× inference throughput at T=8K."]

## Relevance to CHIMERA

[Why this paper affects CHIMERA's design. E.g.: "Mamba-2 is the production
mode-1 mixer; the SSD interface (chunked scan + step) is what ToySSM is
shimmed against."]

## Cited by

- `chimera/...`
- `docs/decisions/ADR-...`
