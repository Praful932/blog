---
title : When You Quantize a VLM, What Happens to What It Sees?
header :
    teaser : /assets/images/blog-5-latentlens-quant-exp/teaser.png
tags:
    - experiment
    - code
excerpt : "Under quantization how does the interpretability of these visual tokens change?"
classes : "wide"
---
## Background

[LatentLens](https://arxiv.org/abs/2602.00462) is a method used to interpret what *visual token representations* encode in an LLM or put simply deciphering what the LLM sees when it processes a token which is part of an image. This is done by encoding a large text corpus and storing contextualized token representations for each token in that corpus. Visual token representations are then compared to these contextualized representations and the top-k nearest neighbor representations serve as descriptions of the visual token.

In this blog I attempt at answering the question : ***Under quantization(LLM-only) does the interpretability of these visual tokens change? If so, how?***
More concretely I have the following hypothesis:

> **H1** — *After LLM quantization, the words corresponding to visual tokens shift from hyponyms towards hypernyms (eg - apple → fruit)*

> **H2** — *% of interpretable tokens degrades in later layers after quantization*

*The original paper only studied FP16, this blog studies what happens to VLMs under different methods of quantization.*

<figure style="width:700px;" class="align-center">
  <img src="/assets/images/blog-5-latentlens-quant-exp/method.png" alt="Illustration of LatentLens (Figure 2 from the paper)">
  <figcaption>Illustration of LatentLens (Figure 2 from the <a href="https://arxiv.org/pdf/2602.00462">paper</a>). LatentLens deciphers that the red-marked visual token is actually a clock/tower (description).</figcaption>
</figure>

### Background for the hypothesis

Quantization techniques reduce memory and computational costs by representing model weights in lower precision (for eg - 8-bit integers). Quantization is generally done for the LLM part of a VLM and the vision encoders are left as-is as they are comparatively smaller and has lesser computation overhead.

However it also comes with a tradeoff, quantization replaces high-precision weights and activations with lower precision integers (8, 4 bits), this introduces quantization error which perturbs the learned function and can change how internal representations can evolve during the forward pass. Literature has shown that this is task specific and the decrease in task performance is especially more for smaller models.

Recent work ([Through a Compressed Lens by Wang et al. 2025](https://arxiv.org/pdf/2505.13963v1)) has shown that quantization degrades explainability not in a uniform sense but is a function of method x quantization technique x model size. It also shows that quantization preferentially damages the final layers more as the later layers are most sensitive to precision changes because small changes there propagate directly to the output distribution. This begs the question do these small changes cause interpretability of visual tokens to also drop at later layers? (H2)

[Related work](https://huggingface.co/blog/embedding-quantization) on embedding quantization has shown that quantization preserves global semantic structure while increasingly distorting local distinctions. Does it have any effect on the phrases found out using this method for the visual tokens? (H1)

### TLDR

1. Quantization method matters more than bit-width alone. GPTQ INT4 (81.75%) preserves how visual tokens are represented far better than BnB INT8 (61.1%) when compared against FP16 (top-1 exact match).
2. Degradation, when it appears is localised and mid-layer not late layer.
3. There is -10% drop in interpretability at layer 16 in BnB NF4 due to degenerate token collapse, although not statistically significant : n= 100, p=0.112 (McNemar).

---

## Experiment Setup

<p align="center"><img src="/assets/images/blog-5-latentlens-quant-exp/image.png" alt="Experiment setup" style="width:500px;"></p>

**Model** - Qwen2-VL-7B-Instruct

**Quantization Arms** - For the same model, quantization methods to assess:

<div style="margin-left: 2em;" markdown="1">

| **Arm** | Method | Precision |
| --- | --- | --- |
| FP16 | - | Full Precision baseline |
| BnB INT8 | BitsAndBytes | 8-bit |
| BnB NF4 | BitsAndBytes | 4-bit NormalFloat |
| GPTQ INT8 | GPTQ | 8-bit |
| GPTQ INT4 | GPTQ | 4-bit |

</div>


**Index -** The same set [https://huggingface.co/McGill-NLP/latentlens-qwen2vl-embeddings](https://huggingface.co/McGill-NLP/latentlens-qwen2vl-embeddings) from the paper is used, which are contextual text embeddings. The index contains ~300k contextual embeddings (shape `[300836, 3584]`) sourced from Visual Genome captions.

**Data**

- 100 images from the PixMo-Cap validation split - [https://huggingface.co/datasets/Praful932/pixmo-cap-val-100](https://huggingface.co/datasets/Praful932/pixmo-cap-val-100)
- 1 patch per image, with patch coordinates fixed from the Full Precision baseline run (so all arms evaluate the exact same spatial locations)

**Controls:**

- only the LLM backbone is quantized (vision tower + projector + LM head stay FP16)
- Reference Index is used as is for all arms, so any shift is attributed purely to visual token distortion

### Metrics

- n - 100 images, ~100 tokens/layer/arm, 800 rows
- H1 was measured using WordNet `min_depth` on the top-5 retrieved neighbour words. A lower `min_depth` indicates a more abstract high-level concept, so a drift toward lower depth signals hypernym climb.

- H2 was measured using an LLM judge (`GPT 5.5`), the judge setup is similar to the original paper, the judge rates whether the top-5 retrieved words for a visual token are semantically related to the image patch. % interpretable per layer per arm is reported.

## Findings

### H1 Findings

1. For NF4, 3/4 top-1 changes are genuine retrieval changes, for a visual token the NF4 quantized model retrieved a word that the FP16 would never have retrieved in the top-5.
2. At an aggregate level, Among the top 50 where the largest depth drop happened, the vocab shift is a mixture of 2 types of cases
    1. One where the hypernym climb makes sense
        
        
        | FP16 word | NF4 word | Layer | Depth Δ | FP16 sim → NF4 sim | Type | Interpretation |
        | --- | --- | --- | --- | --- | --- | --- |
        | grapefruit | fruits | 27 | −5.0 | 0.623 → 0.470 | specific fruit → category | Textbook hypernym, one rung up the taxonomy |
        | grapefruit | fruit | 1 | −2.6 | 0.247 → 0.297 | specific fruit → category | Same abstraction pattern, consistent across layers |
        | transit | routes | 24 | −3.6 | 0.502 → 0.348 | specific service → general concept | From the transport mode to the abstract path concept |
        | hiking | hill | 24 | −4.5 | 0.348 → 0.116 | specific activity → terrain | From what a person does to the place they're in |
        | waves | ocean | 2 | −1.2 | 0.350 → 0.351 | specific phenomenon → body of water | Part → whole; surface pattern abstracted to scene |
        | sand | beach | 4 | −2.0 | 0.345 → 0.358 | specific material → scene/location | From the stuff to its setting |
    2. Weaker examples where the NF4 word is not a hypernym at all, just a different concept the hidden state happened to land near
        
        
        | FP16 word | NF4 word | Layer | Depth Δ | Why it is NOT a hypernym climb |
        | --- | --- | --- | --- | --- |
        | bull | rubber | 2 | -7.0 | `rubber` bears no taxonomic relationship to `bull`; the hidden state moved to an unrelated concept |
        | cat | garden | 2 | -5.0 | `garden` is not a hypernym of `cat`; likely a scene/context word bleeding into the retrieval (a cat photographed in a garden) |
    3. Manually categorizing all 50 rows in the largest-depth-drop table:
        
        
        | Category | Approx. count | Approx. % |
        | --- | --- | --- |
        | Clean hypernym climb (specific -> category/scene) | 6 | 12% |
        | Lateral drift (different, unrelated concept) | ~18 | 36% |
        | Same top-1 word, depth changes via other top-5 neighbors | ~8 | 16% |
        | Noise / non-word tokens (numbers, URLs, partial words) | ~7 | 14% |
        | Ambiguous / borderline cases | ~11 | 22% |
3. At an aggregate level across all layers, the depth difference is small, and the ***hypernym effect is concentrated in specific layers and specific samples*** and not as a uniform global shift.
    
    
    | Condition | Mean depth (top-5) |
    | --- | --- |
    | FP16 | 7.37 |
    | BnB INT8 | 7.54 |
    | BnB NF4 | 7.38 |
    | GPTQ INT8 | 7.38 |
    | GPTQ INT4 | 7.40 |
4. GPTQ ~ FP16, GPTQ being a calibration based quantization technique preserves retrieval geometry. GPTQ INT8 matches FP16's top-1 word in **97.75%** of cases, GPTQ INT4 in **81.75%**.
    
    <p align="center"><img src="/assets/images/blog-5-latentlens-quant-exp/image-1.png" alt="Top-1 word match % by quantization arm" style="width:700px;"></p>
    

### H2 Findings

1. Interpretability % follows a U-shaped curve, peaks at early and late layers and troughs at the middle layer, this pattern holds across all arms, *this shows that which layer you probe is more important than the underlying quantization method.* 
    
    <p align="center"><img src="/assets/images/blog-5-latentlens-quant-exp/image-2.png" alt="Interpretability % by layer (U-shaped curve)" style="width:700px;"></p>
    
2. When compared against FP16, all quantization arms stay within +- 4% at every layer except for BnB NF4 at layer 16. So to answer the hypothesis, ***the interpretability degradation is present only for NF4, concentrated in the middle layers and not the late layers, although statistically not significant, p=0.112, McNemar.***. One possible explanation for NF4, could be that at mid layer the contextual embeddings get compressed into a space where there is no coherent contextual neighbour, before re-expanding in later layers.
    
    <p align="center"><img src="/assets/images/blog-5-latentlens-quant-exp/image-3.png" alt="Interpretability % vs FP16 baseline per layer" style="width:700px;"></p>
    
3. For BnB NF4, layer 16 visual tokens has a degenerative behaviour where the visual token representations collapse into degenerate token neighbourhoods. This is the majority reason of the interpretability drop for BnB NF4 in layer 16. Instead of words that have some meaning, the entire top-5 fills with numbers or nonsense repeated characters. Few examples:
    
    
    | val_idx | FP16 top-5 | NF4 top-5 |
    | --- | --- | --- |
    | 40 | dog, dog, dog, dog, dog | 1, 2, the, the, the |
    | 57 | sky, skies, sky, skies, skies | 1, doesn, 2, hasn |
    | 72 | stacks, stack, stack, stack, piles | 1, 1, 1, 1, 2 |

### Limitations

1. Statistical Power : At n=100 per layer per arm, no comparison reaches p<0.05, the strongest signal is NF4 layer 16, McNemar p=0.112 which is directional but not confirmed.
2. Whether the layer-16 sensitivity generalises across different VLM architectures or image distributions is untested here.
3. The current setup keeps the index at FP-16 and quantizes the model, so quantized models are asked how close they are to FP16 in FP16’s own reference frame. Does quantizing the reference index as per the model being tested change anything is untested.

## Summary:

1. H2 as originally posed, that interpretability degrades in later layers under quantization was not supported. Across BnB and GPTQ, all arms stay within +- 4 % of FP16 at every layer except for BnB NF4 at layer 16 where interpretability drops due to degenerate token collapse. So the effect exists but is localised to the mid layer, not the late layers, and does not clear statistical significance at n=100 (McNemar p=0.112).
2. H1 was also not supported as a global effect, the hypernym climb is real but only appears in 12% of the largest depth-drop cases. Two NF4 specific effects are worth noting:
    1. Degenerate collapse - hidden states drift to void regions, retrieving number tokens
    2. Hypernym climb - certain examples where NF4 causes hypernym climb
3. Calibration based quantization methods preserve retrieval geometry  stronger than the ones that don’t do calibration. GPTQ >>> BnB.
4. Layer 16 consistently emerges as the most sensitive point, it’s where H2 interpretability drops most, where H1 depth signals are stronger(See [7_result_eda_bnb.ipynb](https://github.com/Praful932/latentlens-quant-exp/blob/main/experiment/working_nbs/7_result_eda_bnb.ipynb)) and where degenerate token collapse is also concentrated. This informs that for a quantized VLM, you don’t need to probe every layer, probing the mid-layer will give the most signal.

### Future Work

1. Test the layer-16 signal at a higher power. A larger n, roughly ~3x the current samples would be sufficient to determine whether the layer-16 effect is real or whether noise in a small sample.
2. Rebuild the index using the same quantized model at query time to separate quantization damage to the reference space from quantization damage to the query side.
3. Test the mid-layer compression on other VLM architectures like LLaVA-family, Molmo.

## Acknowledgements

1. [Benno Krojer](https://bennokrojer.com/) (1st author of the Latentlens paper) for feedback on the experiment setup.
2. Arth Singh for providing OpenAI API credits for the experiment.

Code - [https://github.com/Praful932/latentlens-quant-exp/tree/main/experiment](https://github.com/Praful932/latentlens-quant-exp/tree/main/experiment)

## References:

1. LatentLens - [https://arxiv.org/pdf/2602.00462](https://arxiv.org/pdf/2602.00462)
2. Through a Compressed Lens - [https://arxiv.org/pdf/2505.13963v1](https://arxiv.org/pdf/2505.13963v1)