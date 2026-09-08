# Correction script for the paper source

Work top to bottom. Each item gives **where**, the **exact LaTeX** before and after,
and **why** — because several of these look like preferences and are not: four are
type errors, one is a variable-binding bug, and one contradicts the artefact.

Read with [06_TEXT_REVIEW.md](06_TEXT_REVIEW.md) (what is wrong) and
[07_NOTATION_SCHEME.md](07_NOTATION_SCHEME.md) (the target scheme). This document is
the ordered procedure.

> **Note on the current source.** §2.2 and Algorithm 1 have already been partly
> migrated to `Q` for the bit-width. The migration is incomplete, and the half-applied
> state is *worse* than the original: `N` now appears in the format invariant and in
> the clip bounds while being defined nowhere, and `Q` is simultaneously the
> quantization function and the first component of its own argument. Pass 1 finishes
> the migration; do it before anything else.

---

## Pass 0 — Preamble (5 min)

- [ ] `\input{notation}` from [notation.tex](notation.tex).
- [ ] Copy [check_notation.sh](check_notation.sh) next to the `.tex` and run it now to
      get a baseline count.
- [ ] Run the mechanical half of this document:

      perl fix_notation.pl paper.tex            # dry run: what it would change, and why
      perl fix_notation.pl --apply paper.tex    # rewrites, keeps paper.tex.bak

      It applies only the unambiguous, meaning-preserving renames — the glyph and
      spelling fixes from passes 5, 6.6 and 9 — and then **reports** every judgement
      item, with its reason and line numbers, instead of touching it. It never
      rewrites a claim.

**Why.** Every remaining pass is a rename. A macro cannot be spelled two ways, and the
guard turns "did we get them all?" from a re-read into an exit code.

---

## Pass 1 — §2.2, the fixed-point format (highest defect density in the paper)

### 1.1 The definition sentence and Eq. (3)

**Before**

```latex
PreQ-BMC adopts a signed fixed-point format \(\langle Q,I,F\rangle\), where \(N=I+F+1\),
\(I\) includes the sign and integer bits, and \(F\) is the number of fractional bits.
Given a tensor \(\mathbf{A}\), quantization is defined as
\begin{equation}\label{eq:quantization_signal}
Q(\mathbf{A}, \langle Q,I,F\rangle)=
\mathrm{clip}
\left(
\left\lfloor 2^F\mathbf{A} \right\rceil,
-2^{N-1},
2^{N-1}-1
\right),
\end{equation}
```

**After**

```latex
PreQ-BMC adopts a signed fixed-point format \(\langle Q,I,F\rangle\), where \(I\) is the
number of integer bits, \(F\) the number of fractional bits, and \(Q=I+F+1\) the total
width, the final bit being the sign. Given a tensor \(\mathbf{A}\), quantization is
defined as
\begin{equation}\label{eq:quantization_signal}
q_{\langle Q,I,F\rangle}(\mathbf{A})
=\operatorname{sat}_{Q}\!\left(\left\lfloor 2^{F}\mathbf{A}\right\rceil\right),
\qquad
\operatorname{sat}_{Q}(v)=\operatorname{clip}\!\left(v,\,-2^{Q-1},\,2^{Q-1}-1\right),
\end{equation}
where \(\lfloor\cdot\rceil\) denotes rounding to the nearest integer with ties away from
zero, and the resulting integer is interpreted with scaling factor \(2^{-F}\).
```

**Why — six separate defects in four lines.**

1. **`N` is undefined.** The triple is `⟨Q,I,F⟩`, so `N=I+F+1` and `-2^{N-1}` reference a
   symbol that no longer exists in this paper. This is a leftover from the partial
   migration and it breaks the only equation that defines quantization.
2. **"`I` includes the sign and integer bits" contradicts the `+1`.** If `I` already
   counts the sign bit, then `Q=I+F`, not `I+F+1`. The implementation settles it:
   `src/backends/fixed_point.py:24` — "`integer_bits` excludes the sign bit … the
   invariant is `total_bits == integer_bits + fractional_bits + 1`", enforced by a
   `ValueError` in `__post_init__`. Your reported format `⟨13,4,8⟩` satisfies
   `13 = 4+8+1` and violates every other reading. A reviewer who adds `4+8` finds this
   in ten seconds.
3. **`Q` is the function and its own argument.** `Q(\mathbf A, \langle Q,I,F\rangle)`
   cannot be read. Rename the map to `q`, which is *already* the name used in
   Eq. (5) for input quantization — the same operation. One symbol now does one job,
   and `q` in Eq. (5) stops being undefined.
4. **`\lfloor\cdot\rceil` is never defined.** The sentence that defined
   round-half-away-from-zero was dropped in this revision. Eq. (3) now uses an unusual
   bracket with no gloss, and §4.1 later relies on the same rounding mode
   (`RHAZDiv`) without a definition to point back to.
5. **`clip` and `sat` are the same operation under two names** (Eq. 3 vs Eq. 6).
   Defining `sat_Q` here means Eq. (6) inherits it instead of introducing a second.
6. **The paragraph ends in a comma.** After the equation there is no closing clause;
   the sentence is unterminated. This is the visible symptom of defect 4.

### 1.2 Consequential check

- [ ] `grep -n 'N=I+F\|2\^{N-1}\|sat_{\\mathcal{N}' paper.tex` returns nothing.

---

## Pass 2 — §2.3, adversarial robustness (contains a type error)

**Before**

```latex
given a valid
input domain $\mathcal{X}=\prod_{j}[\ell_j,u_j]$, an input $x\in\mathcal{I}$, and a
radius $\varepsilon$, we take $\mathcal{B}_{\varepsilon}(x)=\prod_{j}[\max(\ell_j,
x_j-\varepsilon),\,\min(u_j,x_j+\varepsilon)]$, and say that $\mathcal{\hat N}$ is locally robust at
$x$ with class $c$ if
\[
\forall x'\in\mathcal{B}_{\varepsilon}(x):\quad \widehat{\mathcal N}_{\mathrm{fp}}(x')>\max_{k\neq c}f_k(x'),
\]
```

**After**

```latex
PreQ-BMC adopts the clipped \(L_\infty\) ball as its perturbation model. Let
\(\mathcal X=\prod_j[x_j^-,x_j^+]\) be the valid input domain, \(\mathbf x_0\in\mathcal X\)
an input, and \(\varepsilon\) a radius. We take
\[
B_{\varepsilon}(\mathbf x_0)=\prod_j\left[\max(x_j^-,x_{0j}-\varepsilon),\ \min(x_j^+,x_{0j}+\varepsilon)\right],
\]
and say that a classifier \(g\), with logit vector \(g(\mathbf x)\), is locally robust at
\(\mathbf x_0\) with target class \(t\) if
\begin{equation}\label{eq:local-robustness}
\forall \mathbf x\in B_{\varepsilon}(\mathbf x_0):\quad
g_t(\mathbf x)>g_j(\mathbf x)\quad\forall j\neq t.
\end{equation}
Section~\ref{sec:methodology} instantiates \(g\) with the deployed fixed-point network.
```

**Why.**

1. **Type error: a vector is compared with a scalar.**
   `\widehat{\mathcal N}_{\mathrm{fp}}(x') > \max_{k\neq c} f_k(x')` — the left side has
   no class index, so it is the whole logit vector. As written the inequality is
   ill-typed.
2. **Two different networks in one inequality.** The left side is the quantized network,
   the right side is `f_k`, the floating-point one. Whatever was intended, this states
   that the QNN's output beats the *float* network's runners-up.
3. **`x \in \mathcal{I}` is undefined**, and the domain was named `\mathcal X` in the same
   sentence. Two symbols, one object, eight words apart.
4. **`\mathcal B_\varepsilon` here vs `B_\varepsilon` in Eq. (5)** — different glyph *and*
   different set (this one is clipped, Eq. (5)'s is not). Since Eq. (5) is the property
   the paper claims to verify and the implementation clips, the clipped definition must
   be the one that survives. See Pass 4.2.
5. **`c` here vs `t` from §4.1 onward** for the target class. Standardise on `t`; this
   also frees `c` for the cut bounds in Pass 6.3.
6. **`\ell_j` vs `l`** the layer index — indistinguishable in this font.
7. **Defining robustness for a generic `g`** makes Eq. (5) a literal instance of
   \eqref{eq:local-robustness} rather than a similar-looking restatement. That is worth
   more than the notation fix: it lets §4.1 say "the same property, with `g` the
   deployed network", which is the paper's whole thesis in one sentence.
8. `\mathcal{\hat N}` renders the hat outside the calligraphic box. Use
   `\widehat{\mathcal N}`.

---

## Pass 3 — §2.1 and §2.4 (small, mechanical)

### 3.1 §2.1: `\sigma^{[l]}` vs `\rho_l`

- [ ] In the prose after Eq. (2), replace `$\sigma^{[l]}(\cdot)$ is the activation
      function` with `$\rho_l$ is the activation function of layer $l$`.
- [ ] Add to the same sentence: `$\mathbf z^{[l]}$ is the pre-activation and
      $\mathbf a^{[l]}$ the activation; layer $l$ has $m_l$ neurons.`

**Why.** Eq. (2) already writes `\rho_{l}` and §4.2 writes
`\mathcal G_{\mathrm{act}} = \rho_l(\mathcal G_{\mathrm{pre}})`. `\sigma^{[l]}` appears
once, in the prose that explains Eq. (2) — a second name for the object the equation
just introduced, in the paper's first equation. `m_l` is defined here because Eq. (14)
uses it without definition.

### 3.2 §2.4: the two-valued BMC claim

**Before** — "If the generated constraints are satisfiable, the solver returns a
counterexample demonstrating a property violation; otherwise, the property holds within
the specified bound."

**After** — "If the generated constraints are satisfiable, the solver returns a
counterexample demonstrating a property violation; if they are unsatisfiable, the
property holds within the specified bound. If the solver exhausts its time or memory
budget, the outcome is inconclusive and is treated as neither."

**Why.** §4.9 and §5.1 insist that `Timeout`, `Memout` and `Unknown` "are reported as
inconclusive, never as refutations" — and nine of your ten regions land there. The
background section currently teaches the reader a dichotomy that the contribution
spends a subsection dismantling, and it is the lens through which a skimming reviewer
will read §5. Also: "For QNN" → "For QNNs".

---

## Pass 4 — §4 opening and Eq. (4) (the paper's central formal claim)

### 4.1 The opening paragraph

**Before**

```latex
We call the original ANN as $\mathcal{N}$ and the quantized neural network as ${\hat{\mathcal N}}$.
The layers of the ANN are represented as $f^{[l]}$ and the layers of the QNN as $\hat{f^{[l]}}$.
The preimage of each layer $f^{[l]}$ is represented as $P_i$ and it represents the
under-approximation $P^{2[l]} \subseteq \mathcal N^{1:2[l]}_{2[l]}(\mathcal O)$. Our approach
decomposes this global verification problem into a series of layer-wise counter-example
verification problems. Specifically, for each affine layer $f^{2[l]}$. Initially, we use the
conservative abstract interpretation (DeepPoly) \cite{10.1145/3290354} abstract interpretation
framework to prove that ... admissible output intervals for each layer, $\mathcal{A}^{[l]}_{j}$.
... These intervals are defined by $P^{2[l]}$ ...
```

**After**

```latex
We denote the floating-point network by \(\mathcal N\) and the quantized network by
\(\widehat{\mathcal N}\), with layers \(f^{[l]}\) and \(\widehat f^{\,[l]}\) for
\(l=1,\dots,L\). We write \(\mathcal N^{1:l}=f^{[l]}\circ\cdots\circ f^{[1]}\) for the
prefix of the first \(l\) layers, and likewise \(\widehat{\mathcal N}^{1:l}\) for the
deployed prefix. For each layer \(l\), the backward preimage \(\mathcal P^{[l]}\) is an
under-approximation of the set of layer-\(l\) outputs from which the property still
holds. Our approach decomposes the global verification problem into layer-wise
obligations. We first use DeepPoly~\cite{10.1145/3290354} to prove that the
floating-point network satisfies the property and to compute admissible intervals
\(\mathcal A^{[l]}\) for each layer. We then compute \(\mathcal P^{[l]}\) by backward
MILP and use it to guide synthesis, verifying for each quantized layer
\(\widehat f^{\,[l]}\) the obligations of Theorem~\ref{thm:transfer}.
```

**Why.**

1. **`2[l]` is not a notation, it is a rendering accident.** `P^{2[l]}`,
   `\mathcal N^{1:2[l]}_{2[l]}`, `f^{2[l]}` — a superscript-and-subscript pile that no
   reader can parse and that the rest of the paper (which uses plain `[l]`) does not
   use. One indexing convention, `[l]`, `l = 1..L`.
2. **`P_i` and `P^{2[l]}` are the same object in consecutive sentences.**
3. **`\mathcal N^{1:2[l]}(\mathcal O)` is backwards.** A *preimage* is an inverse image.
   As written, the network is applied *forward* to an output region, which is not what
   `P` is. Either write `(\mathcal N^{1:l})^{-1}(\mathcal O)` or, as above, describe it
   in words and drop the malformed formula.
4. **"Specifically, for each affine layer \(f^{2[l]}\)." is a sentence fragment** — no
   verb, no main clause.
5. **"the conservative abstract interpretation (DeepPoly) … abstract interpretation
   framework"** says "abstract interpretation" twice inside one noun phrase.
6. **"We call the original ANN as \(\mathcal N\)"** is not grammatical English; "call X
   Y" or "denote X by Y", not "call X as Y".
7. **`\mathcal A^{[l]}_{j}`** carries a `j` subscript that appears once and never again.
8. **`\hat{f^{[l]}}`** puts the hat over the entire box including the superscript, so it
   renders displaced up and to the right. Use `\widehat f^{\,[l]}`.
9. **"counter-example"** — spelled closed everywhere else in the paper.

### 4.2 Eq. (4) → a theorem

**Before**

```latex
\begin{equation}\label{eq:main_preposition_milp}
\hat{\mathcal{N}}^{1:2[l]}_{2[l]}(\mathcal{I}) \subseteq P^{2[l]} \implies \hat{\mathcal{N}} \models \langle \mathcal{I}, \mathcal{O} \rangle
\end{equation}
```

and the preceding sentence "By confirming this property at every layer, we establish
that the entire \(\hat{\mathcal N}\) satisfies the global robustness property."

**After**

```latex
\begin{theorem}[Deployed transfer]\label{thm:transfer}
Let \(\widehat{\mathcal X}=q(B_\varepsilon(\mathbf x_0))\). If
\begin{align}
&\widehat{\mathcal X}\subseteq\mathcal A^{[1]},
  &&\text{(base case)}\label{eq:base}\\
&\forall l\le L,\ \forall\widehat{\mathbf a}\in\mathcal A^{[l]}:\ \widehat{\mathbf z}^{[l]}(\widehat{\mathbf a})\in\mathcal G^{[l]}_{\mathrm{pre}},
  &&\text{(layer contracts)}\label{eq:contracts}\\
&\forall l<L:\ \mathcal G^{[l]}_{\mathrm{act}}\subseteq\mathcal A^{[l+1]},
  &&\text{(composition)}\label{eq:composition}\\
&\Psi_j \text{ holds for every } j\neq t,
  &&\text{(output proof)}\label{eq:output}
\end{align}
then \(\widehat{\mathcal N}\) satisfies \eqref{eq:deployed-property}.
\end{theorem}
```

**Why — this is the most important correction in the document.**

1. **As printed, Eq. (4) is false, and your own §4.7 says so.** It asserts that
   containment at a single layer implies the global property. §4.7 opens: "The last
   hidden-layer contract alone does not establish classification." §4.2 adds the
   composition obligation. So the paper states a sufficient condition in §4 and then
   spends §4.2 and §4.7 explaining that it is not sufficient. A formal-methods reviewer
   reads Eq. (4) first.
2. **The quantifier `∀l` is missing** even for the weaker reading.
3. **The base case is missing from the whole paper.** §4.2 gives the inductive step
   (8) and the composition step (9) and never anchors `l = 1`. Without
   \eqref{eq:base}, the chain proves something about `\mathcal A^{[1]}`, not about the
   input region. This is a genuine gap in the argument, not a presentation issue.
4. **`\mathcal I` collides** with the integer-bit count `I` and is undefined; the pair
   `\langle\mathcal I,\mathcal O\rangle` is never destructured, so it does not need to
   be a pair — name the property once and use `\eqref{eq:deployed-property}`.
5. **`\langle\cdot\rangle` now has three jobs** (format, contract, property). Reserving
   it for formats costs nothing.

Stating this as a theorem is also the single highest-value addition available for a
dependability venue: it turns a workflow description into a claim with premises.

---

## Pass 5 — §4.1, Eqs. (5) and (6)

### 5.1 Eq. (5)

- [ ] `\widehat{\mathcal N}_{\mathrm{fp}}(q(\mathbf x))_t` → `\widehat y_t(q(\mathbf x))`,
      and add before the equation: "write \(\widehat{\mathbf y}(\widehat{\mathbf x})
      =\widehat{\mathcal N}(\widehat{\mathbf x})\) for the deployed integer logit vector".
- [ ] Delete `B_{\varepsilon}(\mathbf x_0)=\{\mathbf x:\|\mathbf x-\mathbf x_0\|_\infty\le\varepsilon\}`
      and replace with a pointer to the clipped definition in §2.3.
- [ ] Add: "\(q=q_{\langle Q_0,I_0,F_0\rangle}\) is the input quantizer of
      Eq.~\eqref{eq:quantization_signal}, and
      \(\widehat{\mathcal X}=q(B_\varepsilon(\mathbf x_0))\); since \(q\) is monotone
      coordinatewise, \(\widehat{\mathcal X}\) is exactly the image of the ball, not an
      over-approximation."

**Why.**

1. **`\mathrm{fp}` means fixed-point here and floating-point everywhere else** in the
   paper. This is the one subscript a reader is guaranteed to expand wrongly. Deleting
   it is better than renaming it, because `\widehat y` is needed anyway: Eq. (18)
   already uses `\widehat y_t` without ever defining it. One definition fixes both.
2. **Two incompatible `B_\varepsilon`.** §2.3's is clipped to the valid domain; this one
   is not. They differ whenever the ball leaves `[0,1]^d`, which at `\epsilon = 4` in raw
   pixel units is most pixels — i.e. in four of your ten regions. The implementation
   clips, so the unclipped definition is the one to delete.
3. **The `q(B_\varepsilon)` step is the bridge between the real-valued property and the
   integer harness**, and it is currently unstated. It is also the base case of
   Theorem~\ref{thm:transfer}, so it has to be written down somewhere.

### 5.2 Eq. (6)

- [ ] `\operatorname{sat}_{\mathcal{N}_l}` → `\operatorname{sat}_{Q_l}` (both occurrences,
      equation and prose).
- [ ] After the equation add: "Here \(\widehat W^{[l]}\) and \(\widehat b^{[l]}\) are
      quantized at scale \(S_l=2^{F_l}\) and \(\widehat{\mathbf a}^{[l-1]}\) at
      \(S_{l-1}\), so the product carries scale \(S_lS_{l-1}\) and the rescaling by
      \(2^{F_{l-1}}\) restores scale \(S_l\) before the bias is added."
- [ ] `\operatorname{RHAZDiv}` → `\operatorname{rhaz}`, and use the same name in §4.4
      (which currently says "RHAZ rescaling").

**Why.** `\mathcal N_l` is *the network symbol with a layer subscript* being used for a
bit-width — the collision is now explicit in the source. And Eq. (6) only type-checks
under a specific assignment of formats to `Ŵ`, `b̂`, `â` that the paper never states;
without it the reader cannot verify that the `2^{F_{l-1}}` divisor is the right one, and
the ratio `S_l/S_{l-1}` in Eq. (10) has no derivation to rest on.

---

## Pass 6 — §4.2–§4.7

### 6.1 Eq. (8): apply the function

- [ ] `\widehat{\mathbf z}^{[l]} \in \mathcal G^{[l]}_{\mathrm{pre}}` →
      `\widehat{\mathbf z}^{[l]}(\widehat{\mathbf a}^{[l-1]}) \in \mathcal G^{[l]}_{\mathrm{pre}}`

**Why.** As written, `ẑ` is free: the sentence quantifies over `â` and then constrains a
variable that has nothing to do with it. Eq. (13) already writes
`\widehat{\mathbf z}^{[l]}(\widehat{\mathbf a})` correctly, so this is also an internal
inconsistency between the obligation and the formula that discharges it.

### 6.2 Eqs. (10)–(11): the error budget

- [ ] Replace Eq. (11) with
      `M_j=\max\left(|a_j^-|,\,|a_j^+|\right)`, where `[a_j^-,a_j^+]` is coordinate `j`
      of `\mathcal A^{[l]}`, **in input integer ULPs**.
- [ ] Add after "The first term bounds weight-quantization error": "A real weight rounded
      to \(\widehat W_{ij}/S_l\) carries at most \(1/(2S_l)\) real error; multiplying by
      \(M_j/S_{l-1}\) and converting back to output ULPs multiplies by \(S_l\), so
      \(S_l\) cancels and the term is independent of \(i\)."
- [ ] Add: "\(\boldsymbol\delta^{[l]}\in\mathbb N^{m_l}\) is a vector with components
      \(\delta_i^{[l]}\), all in output ULPs; \(\delta_t\) and \(\delta_j\) in
      Eq.~(17) are components of \(\boldsymbol\delta^{[L]}\)."

**Why.** This is the paper's only derived bound and it currently reads as if it has two
errors: the left side carries an output-neuron index `i` that the first right-hand term
does not have, and a term described as *weight*-quantization error is divided by the
*input* scale `S_{l-1}`. Neither is an error —
`src/synthesis/preqbmc.py:2484-2515` derives exactly this — but only because `M_j` is in
input integer ULPs, which the paper never says. Two sentences convert an apparent
mistake in your central bound into a derivation. The over/underbar notation is also
used only here and never defined; the `\pm` superscript form matches `\mathbf p^\pm` in
Eq. (12) and `g^\pm` in Eq. (17).

### 6.3 Eq. (13): sets are not predicates

- [ ] `\mathcal A^{[l]}(\widehat{\mathbf a}) \land \neg\mathcal G^{[l]}_{\mathrm{pre}}(\widehat{\mathbf z}^{[l]}(\widehat{\mathbf a}))`
      → `\widehat{\mathbf a}\in\mathcal A^{[l]} \ \land\ \widehat{\mathbf z}^{[l]}(\widehat{\mathbf a})\notin\mathcal G^{[l]}_{\mathrm{pre}}`

**Why.** `\mathcal A` and `\mathcal G` are introduced as *boxes* and used with `∈` and
`⊆` in Eqs. (7)–(9), (12), (18). Here alone they are applied as predicates. Pick the
membership form: it is what the harness encodes, and it keeps the calligraphic symbols
meaning exactly one thing.

### 6.4 Eq. (14): block-wise

- [ ] `B_1,\ldots,B_k` → `\mathcal J_1,\ldots,\mathcal J_k`; `i\in B_r` → `i\in\mathcal J_r`.
- [ ] `G_i(x)` → `\gamma_i(\mathbf x)`, and define: "\(\gamma_i(\mathbf x)\) is the
      guarantee for output neuron \(i\)".
- [ ] Add: "The blocks have size \(\beta\), except possibly the last; \(\beta=0\) denotes
      the unpartitioned layer."

**Why.** `B` is the perturbation ball in §2.3 and §4.1 — using it for blocks means the
same letter denotes an input region and an index set two pages apart. `G_i` is a
predicate while `\mathcal G` is a set (see 6.3). And **`\beta` is never defined
anywhere in the paper** although `\beta=2` and `\beta=0` carry RQ3, the RQ2 discussion,
and the entire future-work paragraph. That is not a notation preference: five claims
currently reference an undefined quantity.

### 6.5 Eqs. (15)–(16): relational cuts

**After**

```latex
\begin{equation}\label{eq:cut}
c_k^{[l],-}\ \le\ \mathbf d_k^{[l]}\cdot\widehat{\mathbf a}^{[l-1]}\ \le\ c_k^{[l],+},
\end{equation}
...
\begin{equation}\label{eq:cut-validation}
\forall\widehat{\mathbf x}\in\widehat{\mathcal X}:\quad
c_k^{[l],-}\ \le\ \mathbf d_k^{[l]}\cdot\widehat{\mathbf a}^{[l-1]}(\widehat{\mathbf x})\ \le\ c_k^{[l],+},
\end{equation}
```

with, in the prose: `\(\widehat{\mathbf a}^{[l-1]}(\widehat{\mathbf x})
=\widehat{\mathcal N}^{1:l-1}(\widehat{\mathbf x})\)` is the exact deployed prefix; the
directions are structural — contract cuts take
`\(\mathbf d_k^{[l]}=\widehat W_k^{[l]}\)`, the quantized weight row of the neuron being
constrained, and margin cuts take the difference row
`\(\mathbf d_k^{[L]}=W_j^{[L]}-W_t^{[L]}\)`.

**Why.**

1. **`D_k` and `\widehat{\mathcal X}` are undefined**, and `D_k` reads as an arbitrary
   direction chosen by an unstated heuristic — which makes §4.6 look like a hack. It is
   not one: `src/verification/c_templates.py:216` ("Every cut direction is exactly the
   corresponding quantized weight row") and `src/synthesis/preqbmc.py:3503`
   (`direction = real_weights[competitor] - real_weights[target]`). Naming the two
   families turns an undefined symbol into a design decision, and the margin family
   explains *why* §4.6 exists at all: the difference-row cut bounds precisely the
   quantity whose "independently computed extrema" defeat Eq. (17). Right now §4.6 and
   §4.7 read as unrelated subsections when one is the remedy for the other.
2. **`(\widehat{\mathbf x})` is doing all the work in Eq. (16) and is invisible.** (15)
   constrains a free vector in a box; (16) constrains the image of the input region
   under the deployed prefix. That substitution is the entire content of the
   obligation and the reason a validated cut "does not exclude any reachable deployed
   hidden state". Write the prefix out.
3. **The cut index has no layer.** `\mathcal C` in Eq. (18) is a conjunction of cuts from
   some layer; without `[l]` the reader cannot tell which.
4. Worth one added sentence: the MILP proposal is widened by the inherited budget
   `\boldsymbol\delta^{[l-1]}` and by coefficient-quantization error before ESBMC
   validates it (`c_templates.py:1152`). That is what makes propose-then-certify sound,
   and it is exactly the trust separation §4.9 claims.

### 6.6 Eq. (17): the analytical margin test

- [ ] `G_t^- - \delta_t > G_j^+ + \delta_j` → `g_t^- - \delta_t > g_j^+ + \delta_j`, and
      define: "\([g_j^-,g_j^+]\) is coordinate \(j\) of \(\mathcal G^{[L]}_{\mathrm{pre}}\)".
- [ ] State which network `\mathcal G^{[L]}` bounds.

**Why.** Italic `G` here vs calligraphic `\mathcal G` two subsections earlier, for what is
presumably the same object; and `G_t^-` is undefined. More consequentially: if
`\mathcal G^{[L]}` already bounds the *deployed integer* logits — which is what ESBMC
discharges — then the implementation error is inside it and the `\pm\delta` correction
is either unmotivated or double-counting. As written the reader cannot tell whether the
test is conservative. One clause settles it.

### 6.7 Eq. (18): **a variable-binding bug**

**Before**

```latex
\Psi_j \equiv \forall \widehat{\mathbf h} \in \mathcal A^{[L]}\cap\mathcal C, \quad
\widehat y_t(\widehat{\mathbf a}^{[l-1]}) > \widehat y_j(\widehat{\mathbf a}^{[l-1]}),
```

**After**

```latex
\Psi_j \equiv \forall\,\widehat{\mathbf a}^{[L-1]}\in\mathcal A^{[L]}\cap\mathcal C^{[L]}:\quad
\widehat y_t(\widehat{\mathbf a}^{[L-1]}) > \widehat y_j(\widehat{\mathbf a}^{[L-1]}),
```

**Why.** The quantifier binds `\widehat{\mathbf h}` and the body never mentions it; the
body instead refers to `\widehat{\mathbf a}^{[l-1]}`, which is **free** in this formula
and carries index `l`, not `L`, in a subsection where `l` is not in scope. So the
obligation as printed quantifies over nothing and constrains an undefined variable.
This is not a typographic problem — it is the paper's output-layer proof obligation, and
it does not currently say anything. Binding `\widehat{\mathbf a}^{[L-1]}` directly also
removes `\widehat{\mathbf h}` from the paper (it was never defined), and makes the
membership `\in\mathcal A^{[L]}` visibly the same object that Eq. (9) chains into.

---

## Pass 7 — Structure and misplacement

### 7.1 The research questions are orphaned

The RQ `itemize` sits at the end of §4, after §4.9, separated by a bare `%-----` rule,
with no heading and no introductory sentence tying it to §4.

- [ ] Move the whole block to §5, immediately after "This section presents the results…",
      or give it a `\subsection{Research Questions}` at the end of §4.

**Why.** As placed, it reads as a continuation of the synthesis-algorithm subsection.
The questions are answered in §5 and belong adjacent to their answers; a reader
following the RQ thread currently has to jump backwards across a section boundary.

### 7.2 Three duplicated passages

- [ ] **Introduction ¶3–¶4 restate ¶1–¶2.** ¶1 (last sentence) + ¶2 make two claims —
      "embedded deployment motivates compression, quantization is the most suitable" and
      "quantization is not semantics-preserving, here are the mechanisms". ¶3 and ¶4
      make the same two claims again, at greater length, citing the same references.
      Delete ¶3–¶4 and keep the mechanism list once.
- [ ] **§5.1, last paragraph** repeats the last sentence of the paragraph above it: "no
      claim in this paper extends beyond that class" then "should not be interpreted as
      evidence of scalability beyond this model class". Keep one.
- [ ] **Conclusion has two future-work openings.** "Future work will extend the set of
      supported model classes and evaluate PreQ-BMC on larger and more diverse neural
      network architectures. Future work follows directly from that failure profile."
      And the final sentence repeats the same point a third time ("extend the evaluation
      beyond feed-forward affine/ReLU networks to larger architectures"). Collapse to
      one paragraph: the two missing arms first, then the model-class extension.

**Why.** Beyond the reading experience: the Introduction duplication costs roughly half
a column, which is what Theorem~\ref{thm:transfer} and the §2.3 rewrite need. This is
where the space comes from.

### 7.3 Algorithm 1 vs Figure 1

- [ ] Algorithm 1 has no path for "no candidate format remains"; Figure 1 has `FAILED`.
      Add an `\Else \Return \textsc{Failed}` after the candidate loop.
- [ ] Figure 1 shows three margin tiers (analytic / output harness + cuts / end-to-end);
      Algorithm 1 shows only the first two. Either add Tier 3 or remove it from the
      figure.
- [ ] Status vocabulary: Figure 1 says `VERIFIED`, Algorithm 1 says `DeployedTransfer`,
      §5 says "deployed-transfer certificate", and the implementation emits
      `guarantee_level: deployed-transfer`. Use the table in
      [04_NOTATION_AND_TEXT_FIXES.md](04_NOTATION_AND_TEXT_FIXES.md).
- [ ] Figure 1's `bit-jump`, `no-sat` and `.so` appear nowhere in the body — replace per
      [07_NOTATION_SCHEME.md §5](07_NOTATION_SCHEME.md).
- [ ] Line 13, "if a hidden witness **may be** spurious", is not a decidable condition.
      State the actual test.

**Why.** A reviewer checks the figure against the algorithm against the prose. Three
vocabularies for one status set is the most visible symptom of the "lack of
proof-reading" comment.

---

## Pass 8 — Claims that the artefact contradicts

These are not notation. Do them, or the notation work is wasted.

### 8.1 §4 opening: two false claims in one sentence

**Before** — "which aims to find the **optimal** bit-width for each layer of a QNN while
ensuring that the defined bit-widths satisfy robustness properties using the **SMT Theory
of Fixed-Point Arithmetic**."

**After** — "which selects, for each layer, the smallest format from a candidate list
that discharges the layer's obligations, and verifies those obligations bit-precisely
over the generated integer C."

**Why.** (i) Algorithm 1 accepts the *first* candidate that works, per layer, and the
paper defines no objective function — "optimal" has no referent and is not attained.
(ii) §5.1 says ESBMC runs "with Z3 in **bit-vector** mode" on integer C with `__int128`
products and explicit `rhaz`; the fixed-point theory of Baranowski et al. is discussed in
§3 as *related work* and is not the encoding used. Claiming an encoding you do not use is
the kind of error that costs credibility disproportionately at a formal-methods venue.

### 8.2 Table 2 and §5.3: the accuracy figure

- [ ] Table 2 caption: delete "Accuracy is measured on the full test set."
- [ ] Report both sets explicitly: full test set — float32 `90.26\%`, quantized Keras
      `90.30\%`; 100-sample probe set — quantized Keras, Python fixed-point and
      generated C all `94.00\%`.
- [ ] §5.3: delete "obtained `94.00\%` full-test accuracy on the generated C backend,
      whereas Quadapter's accuracy fell from `90.26\%` to `80.23\%`". Replace with the
      comparable pair: Quadapter `90.26 → 80.23` full-test; PreQ-BMC `90.26 → 90.30`
      full-test, with the three backends indistinguishable on the probe set.
- [ ] Same fix in the abstract and in RQ4.

**Why.** Your own audit, `output/cross_tool_comparison/results.tex:400-411`:
"the `94.00\%` shared by quantized Keras, the Python interpreter, and the generated C
program is measured on the 100-sample probe set … **The `94.00\%` figure must not be
differenced against the `90.26\%` baseline** … The full-test accuracy of the deployed C
program was never measured." The paper currently does exactly that, four times, and the
resulting 13.8-point gap is an artefact of comparing two different evaluation sets. An
artefact reviewer opening `table_quality_metrics.csv` finds this immediately.

### 8.3 Conclusion vs RQ1

**Before** — "prediction mismatches and logit deviations reveal numerical effects that
are invisible at the model level."

**After** — "bit-exact agreement between the generated C and the fixed-point reference,
together with a residual \(9.01\times10^{-3}\) logit gap to the float-trained model,
locates the deployment gap between the floating-point and fixed-point semantics rather
than in code generation."

**Why.** RQ1 and §5.3 both report **zero** classification mismatch. The conclusion
asserts mismatches that the results section says did not occur. The replacement is both
true and a sharper point.

### 8.4 Other reporting fixes

- [ ] `Calls` in Table 2: define in the caption as "ESBMC queries executed" and reconcile
      with §6's `19`. The audit: 21 = 19 obligations + 2 vacuity sentinels. Note also
      that cut validations increment the same counter
      (`preqbmc.py:3361`), so say whether they are included.
- [ ] §5.3 "all **six** completed MNIST executions" — the audit says seven (four
      `1blk_10`, three `1blk_25`), with medians `1209.12` and `1209.53`. Reconcile.
- [ ] §5.3 "all six completed MNIST executions **ended in timeout**" — "completed" and
      "timed out" contradict. Use "terminated at the timeout".
- [ ] §5.3 "one **valid** Iris execution" — say why the others were excluded (missing
      `ONNX2C_PATH`, an infrastructure failure, not an `Unknown` verdict). Excluding a
      competitor's runs without stating the reason reads badly.
- [ ] §5.1 states the 20 GiB cap "binds well below machine RAM" without giving the RAM;
      promises "Python, compiler, and solver versions for all tools" and gives only
      ESBMC 7.11.0 and CBC; and refers to "the artifact" with no DOI or URL.
- [ ] §5.1 vs §5: the paper claims five networks across three datasets; §5 reports
      PreQ-BMC results on MNIST only. Report the Iris/Seeds arms or narrow the claim.
- [ ] Different budgets: 300 s per ESBMC query vs CEG4N's 1200 s per run. Say so.
- [ ] §5.3 "This is smaller and substantially faster than PreQ-BMC" — the subject is a
      percentage. Name it: "Quadapter's format is more compact and its run faster".

---

## Pass 9 — LaTeX hygiene

- [ ] `\hat{f^{[l]}}` → `\widehat f^{\,[l]}` (hat currently sits over the superscript).
- [ ] `\mathcal{ \hat{N}}` → `\widehat{\mathcal N}` (leading space inside `\mathcal`).
- [ ] `$\textsc{Memout}$` → `\textsc{Memout}` — `\textsc` in math mode does not produce
      small caps.
- [ ] `20\, GiB` → `20\,GiB` (the space after `\,` adds a second gap).
- [ ] `\epsilon` → `\varepsilon`: five occurrences in §5.2–§5.3 and Table 2's caption,
      against `\varepsilon` in §2.3 and §4.1. Same symbol, two glyphs.
- [ ] Give every displayed equation a `\label`: (7), (11), (12), (13), (14), (15), (16),
      (17) currently have none, so none can be referenced, and the text already refers to
      equations by position ("the condition in equation~\ref{...}").
- [ ] `\begin{figure}[H]` requires `float` and forbids the float from moving; in an LNCS
      two-column layout prefer `[t]`.
- [ ] Rename `main_diagram-Main Diagram (Updated).drawio.pdf` — spaces and parentheses in
      a graphics filename break `\includegraphics` on several TeX distributions.
- [ ] Figure 1 caption "Proposed Methodology Workflow." — title case unlike every other
      caption, and it states nothing. Say what the reader should take from it.
- [ ] Table 2: define "Q-acc." in the caption; explain `n/a` for Quadapter's `Calls`
      (not instrumented) rather than leaving it ambiguous against CEG4N's `1`.
- [ ] References: `\cite{cordeiro2025esop}` is used for ESBMC in §4 while
      `\cite{Gadelha2019}` is used for it in §2.4 — pick one. Reluplex is named in §2 but
      only `\cite{katz2019marabou}` is given. Check the `.bib` for sentence-cased
      acronyms (`smt`, `ansi-c`, `Qnnrepair`), month artefacts (`(01 2004)`), and the two
      truncated DOIs.

---

## Pass 10 — Verify

```bash
perl fix_notation.pl paper.tex         # expect: no mechanical substitutions pending,
                                       #         and an empty MANUAL list
./check_notation.sh paper.tex          # expect: notation: clean
grep -n 'N=I+F\|2\^{N-1}' paper.tex    # expect: nothing
grep -n '2\[l\]\|P_i\|\\mathcal{I}' paper.tex
grep -n 'widehat{\\mathbf h}\|D_k\|\\epsilon' paper.tex
grep -cn 'full test set' paper.tex     # expect: 0 in the Table 2 caption
```

Then read the **built PDF**, not the source. At least three defects in this list —
`\lfloor\cdot\rceil` in Eq. (3), the displaced hat in `\hat{f^{[l]}}`, the `2[l]` pile —
are rendering outcomes that a source-only pass cannot see.

---

## Order and effort

| Pass | Content | Effort | Blocking? |
| --- | --- | --- | --- |
| 1 | §2.2 format | 15 min | Yes — the arithmetic is wrong on the page |
| 8.2 | Accuracy evaluation set | 30 min | Yes — the artefact contradicts the paper |
| 4 | §4 opening + Theorem | 2 h | Yes — the central claim is false as printed |
| 6.7 | Eq. (18) binding | 5 min | Yes — the obligation says nothing |
| 2, 5 | §2.3, Eqs. (5)–(6) | 45 min | High |
| 8.1, 8.3 | Optimality, SMT theory, conclusion | 20 min | High |
| 6.1–6.6 | Remaining equations | 1 h | Medium |
| 7 | Structure, duplication | 1 h | Medium — buys the space passes 4 and 2 need |
| 8.4, 9 | Reporting, hygiene | 1.5 h | Medium |

Passes 1, 4, 6.7 and 8.2 are the ones where the paper currently states something untrue.
Everything else is consistency, and consistency is what the reviewer actually complained
about — but a reviewer who finds an untrue statement stops trusting the consistent parts.
