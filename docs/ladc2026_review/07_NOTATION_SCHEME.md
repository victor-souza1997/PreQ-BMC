# Fixing the notation: one scheme, applied mechanically

The collisions listed in [06_TEXT_REVIEW.md](06_TEXT_REVIEW.md) §2 are not eleven
independent typos. They are the result of assigning symbols locally, section by
section, with no global budget. Proofreading will not fix that — each pass restores
some and breaks others. What fixes it is a stated convention, a symbol table the
paper prints, a macro layer so the source cannot express a violation, and a grep
guard in the build.

Companion files: [notation.tex](notation.tex) (the macro block) and
[check_notation.sh](check_notation.sh) (the guard).

---

## 1. The convention (three lines, goes in §2 as a footnote or preamble)

> Calligraphic uppercase denotes sets and regions (`𝒩`, `𝒳`, `𝒜`, `𝒢`, `𝒫`, `𝒯`,
> `𝒥`, `𝒞`, `𝒬`); italic denotes scalars; boldface denotes vectors; a hat denotes a
> deployed integer-valued object (`𝒩̂`, `â`, `ẑ`, `Ŵ`, `ŷ`). Interval endpoints are
> written with superscript `−` and `+`.

This single rule resolves five of the collisions by construction, because a
calligraphic symbol may then only ever appear next to `∈`, `⊆`, `∩` or `⊕` — which is
also the fix for F13 (sets used as predicates in Eqs. 13 and 14).

## 2. The renames

Each collision is resolved by moving the symbol whose claim is weaker, and in three
cases by deleting a symbol rather than renaming it.

| Collision | Keeps the symbol | Moves / disappears | Why this way round |
| --- | --- | --- | --- |
| `N` network vs. bit-width | `𝒩`, `𝒩̂` (network) | width `N` → **`Q_l`**; `sat_{N_l}` → `sat_{Q_l}` | `Q` is what the implementation already calls it: `fixed_point.py` raises "expected Q={total_bits}" and `table_export.py` emits the column `"Q"`. Paper and code converge for free. |
| `Q` quantizer vs. candidate widths | quantizer → **`q_{⟨Q,I,F⟩}(·)`** | candidate set → **`𝒬`** | `q` is *already* the input quantizer in Eq. (5). Making Eq. (3) `q` unifies two maps that were always the same map, and `𝒬` is a set, so the calligraphic rule places it. |
| `I` integer bits vs. input domain vs. `⟨I,O⟩` | `I_l` (integer bits) | domain → **`𝒳`** (as §2.3 already had it); `⟨I,O⟩` → **`Φ`** | `𝒳` is the paper's own symbol two lines earlier. The property pair is never destructured, so it does not need to be a pair. |
| `B` ball vs. blocks | `B_ε(x₀)` (ball) | blocks → **`𝒥_1,…,𝒥_k`** | Blocks are index sets; `𝒥` for an index set is standard, and the ball is cited far more often. |
| `C` cut bounds vs. cut set | `𝒞` (the conjunction of cuts) | bounds → **`c_k^-`, `c_k^+`** | Lowercase for scalars per the rule. `c` is free once the target class becomes `t`. |
| `c` target class (§2.3) vs. `t` (§4.1) | **`t`** | `c` in §2.3 | §4 onward is where the symbol does work. |
| `f_c` / `𝒩̂_fp(q(x))_t` / `ŷ_t` | **`f_t`** (float logits), **`ŷ_t`** (deployed logits) | `𝒩̂_fp` deleted | Two networks genuinely need two symbols; three notations for two objects do not. `fp` reads as "floating-point" in a paper that says it fifty times — delete it, not rename it. |
| `σ^{[l]}` vs. `ρ_l` | **`ρ_l`** | `σ^{[l]}` | `ρ_l` is what Eq. (2) and §4.2's `𝒢_act = ρ_l(𝒢_pre)` already use. |
| `ℓ_j` domain bound vs. `l` layer index | `l` (layer) | bounds → **`x_j^-`, `x_j^+`** | Not in the original list, but `ℓ` and `l` are indistinguishable in this font, and the superscript form matches the endpoint rule. |
| bars in Eq. (11) vs. `p^±` vs. `G^±` | **superscript `−`/`+`** | `a̲_j`, `ā_j` → `a_j^-`, `a_j^+`; `G_t^-` → `g_t^-` | One endpoint convention, three sites. |

**Deleted rather than defined** — the cheapest fixes in the list:

- **`ĥ`** (Eqs. 15, 16, 18) is the last hidden activation, i.e. `â^{[L−1]}`. Write that.
  One fewer symbol, and it makes the assume-guarantee chain visible in the formula.
- **`D_k`** (Eq. 15) survives as **`d_k^{[l]}`**, but only because it must be *defined*,
  not because it is free. The implementation uses two structural families and the
  paper names neither:
  *contract cuts* take the direction to be the quantized weight row of the neuron
  being constrained (`c_templates.py:216`: "Every cut direction is exactly the
  corresponding quantized weight row"), and *margin cuts* take the **difference row**
  `W_j^{[L]} − W_t^{[L]}` (`preqbmc.py:3503`,
  `src/tests/test_difference_row_margin_bound.py`). Say so: an undefined symbol
  becomes a stated design decision, and the second family explains *why* §4.6 exists
  — the cut bounds the margin direction itself, which is exactly the quantity whose
  independent extrema make Eq. (17) fail.
- **`𝒪`** disappears with `⟨I,O⟩` once the property is named `Φ`.

**Defined where they are first used** — the remainder:

| Symbol | Definition to add | Where |
| --- | --- | --- |
| `m_l` | "layer `l` has `m_l` neurons" | §2.1, with Eq. (1) |
| `Φ` | "write `Φ` for the local-robustness property at `x₀` with radius `ε` and target `t`" | §2.3, after the ball |
| `𝒳̂` | "`𝒳̂ = q(B_ε(x₀))`, the integer input box; `q` is monotone coordinatewise, so this is exact rather than an over-approximation" | §4.1, with Eq. (5) |
| `ŷ` | "`ŷ(x̂) = 𝒩̂(x̂)` is the deployed integer logit vector" | §4.1 |
| `M_j` | "`M_j = max(|a_j^-|, |a_j^+|)` **in input integer ULPs**, where `[a_j^-, a_j^+]` is coordinate `j` of `𝒜^{[l]}`" | §4.3, Eq. (11) — this is also finding F19 |
| `β` | "`𝒥_1,…,𝒥_k` partition the `m_l` output indices into blocks of size `β`, except possibly the last; `β = 0` denotes the unpartitioned layer" | §4.5 — **required**, `β = 0` and `β = 2` appear five times in §5–§6 |
| `γ_i` | "`γ_i(x)` is the guarantee for output neuron `i`" | §4.5, Eq. (14) |
| `δ` convention | "`δ^{[l]} ∈ ℕ^{m_l}` is a vector; `δ_i^{[l]}` its component; `δ_t`, `δ_j` in (17) are components of `δ^{[L]}`. All in output ULPs." | §4.3 |

## 3. The symbol table to print

Half a column in §2, or a table in an appendix. A formal-methods reviewer reads it
first and it is the cheapest possible signal that the notation is under control.

| Symbol | Meaning |
| --- | --- |
| `𝒩`, `𝒩̂` | floating-point network; deployed fixed-point network |
| `f^{[l]}`, `f̂^{[l]}` | layer maps; `L` affine layers, layer `l` with `m_l` neurons |
| `ρ_l` | activation of layer `l`: ReLU for hidden, identity for output |
| `z^{[l]}`, `a^{[l]}` | real pre-activation and activation |
| `ẑ^{[l]}`, `â^{[l]}` | deployed integer pre-activation and activation |
| `f_t(x)`, `ŷ_t(x̂)` | float and deployed logit for class `t` |
| `𝒳`, `𝒳̂` | valid input domain; deployed integer input box `q(B_ε(x₀))` |
| `B_ε(x₀)` | clipped `L∞` ball, `∏_j [max(x_j^-, x_{0j} − ε), min(x_j^+, x_{0j} + ε)]` |
| `Φ` | the local-robustness property at `x₀`, radius `ε`, target class `t` |
| `⟨Q_l, I_l, F_l⟩` | format of layer `l`; `Q_l = I_l + F_l + 1` (one sign bit) |
| `S_l = 2^{F_l}` | scale of layer `l`; one integer ULP is `2^{−F_l}` |
| `q_{⟨Q,I,F⟩}` | quantizer, Eq. (3); `sat_Q` is its clipping component |
| `rhaz(·, 2^F)` | division with round-half-away-from-zero |
| `𝒬` | set of candidate formats searched per layer |
| `𝒫^{[l]}` | MILP backward preimage for the output of layer `l` |
| `𝒜^{[l]}`, `𝒢^{[l]}_pre`, `𝒢^{[l]}_act` | assumption box; guaranteed pre-activation box; `ρ_l(𝒢^{[l]}_pre)` |
| `𝒯^{[l]}` | deflated target, Eq. (12) |
| `δ^{[l]}`, `M_j` | error budget (output ULPs); input magnitude bound (input ULPs) |
| `𝒥_r`, `β`, `k` | output-index blocks; block size; number of blocks |
| `c_k^-`, `c_k^+`, `𝒞` | relational cut bounds; conjunction of validated cuts |
| `γ_i`, `Ψ_j` | per-neuron guarantee; competitor obligation for class `j` |
| `u^-`, `u^+` | lower and upper endpoint of any interval `u` |

## 4. The equations, rewritten

Only the ones that change. `(4)` is replaced outright — see finding F2.

```
(2)   a^{[l]} = ρ_l(z^{[l]})                        [σ deleted from the prose]

(3)   q_{⟨Q,I,F⟩}(A) = sat_Q(⌊2^F A⌉),  Q = I + F + 1
      sat_Q(v) = clip(v, −2^{Q−1}, 2^{Q−1} − 1)
      [⌊·⌉ = ties away from zero; one name for saturation; F1 fixed]

(4)   Theorem. If
        (i)   𝒳̂ ⊆ 𝒜^{[1]}                                  (base case)
        (ii)  ∀l ≤ L, ∀â ∈ 𝒜^{[l]} : ẑ^{[l]}(â) ∈ 𝒢^{[l]}_pre   (layer contracts)
        (iii) ∀l < L : 𝒢^{[l]}_act ⊆ 𝒜^{[l+1]}                (composition)
        (iv)  Ψ_j holds for every j ≠ t                      (output proof)
      then 𝒩̂ ⊨ Φ.
      [replaces a single-layer implication that §4.7 contradicts]

(5)   ∀x ∈ B_ε(x₀) : ŷ_t(q(x)) > ŷ_j(q(x)),  ∀j ≠ t
      [B_ε clipped, as defined once in §2.3; 𝒩̂_fp deleted]

(6)   ẑ_i^{[l]} = sat_{Q_l}( rhaz(Σ_j Ŵ_ij^{[l]} â_j^{[l−1]}, 2^{F_{l−1}}) + b̂_i^{[l]} )
      where Ŵ^{[l]}, b̂^{[l]} are at scale S_l and â^{[l−1]} at S_{l−1},
      so the product carries S_l·S_{l−1} and rhaz restores S_l before the bias.

(7)   contract of layer l:  (𝒜^{[l]}, 𝒢^{[l]}_pre)      [parentheses; ⟨·⟩ = formats only]

(8)   ∀â ∈ 𝒜^{[l]} : ẑ^{[l]}(â) ∈ 𝒢^{[l]}_pre           [ẑ applied to â, matching (13)]

(9)   𝒢^{[l]}_act ⊆ 𝒜^{[l+1]}   for l < L,   and   𝒳̂ ⊆ 𝒜^{[1]}

(10)  δ_i^{[l]} = ⌈ Σ_j M_j / (2 S_{l−1}) ⌉ + 1 + ⌈ (S_l/S_{l−1}) Σ_j |W_ij^{[l]}| δ_j^{[l−1]} ⌉

(11)  M_j = max(|a_j^-|, |a_j^+|),  in input integer ULPs,
      where [a_j^-, a_j^+] is coordinate j of 𝒜^{[l]}.
      [S_l cancels between the weight-rounding error and the ULP conversion, which is
       why the first term carries S_{l−1} and no i index — say so; see F19]

(13)  â ∈ 𝒜^{[l]}  ∧  ẑ^{[l]}(â) ∉ 𝒢^{[l]}_pre          [membership, not predicates]

(14)  ∀x ⋀_{i=1}^{m_l} γ_i(x)  ⟺  ⋀_{r=1}^{k} ∀x ⋀_{i∈𝒥_r} γ_i(x)

(15)  c_k^{[l],−} ≤ d_k^{[l]} · â^{[l−1]} ≤ c_k^{[l],+}
      d_k^{[l]} ∈ ℤ^{m_{l−1}}, c_k^{[l],±} ∈ ℤ, at scale 2^{F_d} · S_{l−1}.
      Contract cuts take d_k^{[l]} = Ŵ_k^{[l]}; margin cuts take
      d_k^{[L]} = W_j^{[L]} − W_t^{[L]} (the difference row). [ĥ gone; D_k defined]

(16)  ∀x̂ ∈ 𝒳̂ : c_k^{[l],−} ≤ d_k^{[l]} · â^{[l−1]}(x̂) ≤ c_k^{[l],+},
      where â^{[l−1]}(x̂) = 𝒩̂^{1:l−1}(x̂) is the exact deployed prefix in the
      accepted formats of layers 1..l−1 — not a box variable. That substitution is
      the entire content of the obligation: (15) constrains a free vector, (16)
      constrains the image of the input region under the deployed prefix.
      [`preqbmc.py:3307` "Validate a MILP-proposed cut over the exact deployed
       prefix with ESBMC"]

(17)  g_t^- − δ_t > g_j^+ + δ_j
      where [g_j^-, g_j^+] is coordinate j of 𝒢^{[L]}_pre.
      [and state which network 𝒢^{[L]} bounds, or the ±δ is unmotivated — F06/F17]

(18)  Ψ_j ≡ ∀â ∈ 𝒜^{[L]} ∩ 𝒞 : ŷ_t(â) > ŷ_j(â)         [ĥ → â^{[L−1]} ∈ 𝒜^{[L]}]
```

## 5. Figure 1 and the status vocabulary

Three strings in the figure appear nowhere in the body: `bit-jump`, `no-sat`,
`witness replays Python & .so?`. Replace with the body's own words:

| Figure 1 today | Replace with |
| --- | --- |
| `counterexample → bit-jump` | `counterexample → next candidate in 𝒬` |
| `⊧ contract ∧ no-sat` | `contract holds ∧ assumption non-vacuous` |
| `witness replays Python & .so?` | `witness replays in the reference interpreter and the compiled backend?` |
| `COUNTER-EXAMPLE FOUND` | `COUNTEREXAMPLE` (the body spells it closed everywhere but §4) |

The status names themselves are a separate defect with its own table in
[04_NOTATION_AND_TEXT_FIXES.md](04_NOTATION_AND_TEXT_FIXES.md#48--figure-1--algorithm-1--status-vocabulary-t13-a11);
apply that one at the same time, since both edits touch the same figure.

## 6. Enforcement

Manual consistency does not survive three co-authors and a rebuttal deadline. Two
mechanisms, both cheap:

**Macros.** Copy [notation.tex](notation.tex) into the preamble and use `\qnet`,
`\asm{l}`, `\fmt{l}`, `\blk{r}` rather than raw glyphs. A macro cannot be spelled
inconsistently, and renaming later is one line. Every symbol in the table above has
one.

**A guard.** [check_notation.sh](check_notation.sh) greps the source for the retired
spellings — `\mathcal{I}`, `_{\mathrm{fp}}`, `\sigma^{[`, `2[l]`, `D_k`, `\hat{h}`,
`\epsilon` (should be `\varepsilon`), `counter-example`, `N = I + F`, bare `Q(`,
`\bar{a}`/`\underline{a}`, `sat_{N` — and exits non-zero. Run it in the build, or as
a pre-commit hook on the paper repo. It takes under a second and it catches exactly
the class of defect the reviewer called out.

## 7. Order

1. Preamble: add `notation.tex`, add the guard.
2. §2: convention footnote, symbol table, and the `Q_l = I_l + F_l + 1` fix (F1).
   §2 is where most of the renames land.
3. §4.1–§4.3: Eqs. (5), (6), (10), (11) — the `fp` deletion and the ULP units.
4. §4: replace Eq. (4) with the theorem (F2). This is a content edit, not notation,
   but it must happen before §4.7 reads as a contradiction.
5. §4.5–§4.7: `β`, `𝒥_r`, `γ_i`, the deletion of `D_k` and `ĥ`.
6. Figure 1 and Algorithm 1 together, with the status table.
7. Run the guard; then read the *built PDF*, not the source — several of these are
   rendering defects (`⌊·⌉` in Eq. 3) that a source-only pass cannot see.
