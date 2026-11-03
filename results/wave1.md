# Wave 1 results

Official score is category-macro CER from `badger-scribe-data/metric.py`.
Lower is better. Wall-clock is device-specific; do not mix Mac and 3060 times in one ranking.

## Holdout (val)

| run_id | model | device | dtype | meta | macro CER | Kade CER | Dominy CER | Survey CER | human CER | median CER | WER | sec/page | peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| churro3b-zeroshot | churro | cuda | bf16 | False | 0.3189 | 0.2698 | 0.3743 | 0.3125 | 0.2698 | 0.1698 | 0.4577 | 12.24 | 7401 |
| churro3b-zeroshot-meta | churro | cuda | bf16 | True | 0.3938 | 0.3135 | 0.4442 | 0.4237 | 0.3135 | 0.1799 | 0.5005 | 14.59 | 7402 |
| qwen25vl-3b-zeroshot | qwen2.5-vl-3b | cuda | bf16 | False | 0.4585 | 0.5564 | 0.4861 | 0.3331 | 0.5564 | 0.4825 | 0.8347 | 11.40 | 7401 |
| qwen25vl-7b-zeroshot | qwen2.5-vl-7b | cuda | 4bit | False | 0.3447 | 0.5129 | 0.4023 | 0.1188 | 0.5129 | 0.4493 | 0.7674 | 11.79 | 6041 |
| qwen3vl-8b-zeroshot | qwen3-vl-8b | cuda | 4bit | False | 0.4461 | 0.6055 | 0.6185 | 0.1142 | 0.6055 | 0.4956 | 0.8317 | 31.22 | 6557 |
| olmocr-2-7b-zeroshot | olmocr-2-7b | cuda | 4bit | False | 0.2956 | 0.4873 | 0.3216 | 0.0779 | 0.4873 | 0.4392 | 0.7721 | 8.57 | 6041 |
| pipeline-kraken-trocr | pipeline-kraken-trocr | cuda | fp16 | False | 0.8930 | 0.9668 | 0.8319 | 0.8803 | 0.9668 | 0.9694 | 0.9938 | 0.37 | — |

## Smoke (12 val pages)

| run_id | model | device | dtype | macro CER | Kade CER | human CER | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-churro-meta | churro | cuda | bf16 | 0.3310 | 0.1305 | 0.1305 | ok |
| smoke-qwen25vl-3b | qwen2.5-vl-3b | cuda | bf16 | 0.4441 | 0.4276 | 0.4276 | ok |
| smoke-qwen25vl-7b | qwen2.5-vl-7b | cuda | 4bit | 0.3190 | 0.4421 | 0.4421 | ok |
| smoke-qwen3vl-8b | qwen3-vl-8b | cuda | 4bit | 0.4542 | 0.4129 | 0.4129 | ok |
| smoke-olmocr-2-7b | olmocr-2-7b | cuda | 4bit | 0.3059 | 0.4335 | 0.4335 | ok |
| smoke-deepseek-ocr | deepseek-ocr | cuda | bf16 | 0.9289 | 1.0000 | 1.0000 | ok |
| smoke-pipeline-trocr | pipeline-kraken-trocr | cuda | fp16 | 0.9056 | 0.9516 | 0.9516 | ok |

## LoRA base

**Pick: `churro`** (`stanford-oval/churro-3B`), run `churro3b-zeroshot`. Best human-label mean CER on the frozen holdout (0.2698); Kade CER 0.2698. Fits the intended 12 GB / 24 GB deploy path if the winning dtype did.

## Qualitative (worst pages from `churro3b-zeroshot`)

### dominy_accounts
- `dominy_008_p004` CER 1.0: pred `Abroad Brought over from Page 2d 52 19
19 an eye in fame 13
24,4 spirel & nett 2 pin thorns 52 82
Aug 14 1857 Credit for balance on account 987. 52 82
30 for a `
- `dominy_001_p003` CER 1.0: pred `Hr. v. d. Lanen
vulk auctot & brevet
59x9.7/1`

### kade_letters
- `kade_007_p162` CER 1.0: pred `Vomtag2 haben wir uns gemacht
u. das Bierfleisch u Fett ge=
schritten, u. Montag vormittag
mahlen lassen, u. bis halb
fünf war alles geschafft, daß
ging schnell`
- `kade_007_p151` CER 1.0: pred `Jnean, Wiz.
Dec. 6-1937
Lieber Sohn!
Habe diesen Brief heute
nachmittag, u. wann ich irgendwo
schreiben kann, von Morgens ist ja
so viel Arbeit. Wie wollen
Sams`

### survey_notes
- `survey_001_p0030` CER 1.0: pred `Township 4 North, Range 1
Chain North between Section
17 and 18
Variation 7° 10' East
40.00 Left Quarter Section post
Hemlock 8 N 10 W 25 links
Hazel 108 65 E 2`
- `survey_001_p0029` CER 1.0: pred `Township 4 North, Range 1
Chains East Random between
Sections 17 + 20
Variation 1° East
79.75
Intersect North & South
line 30 links North
of post.
West of 4th P`


Go/no-go: LoRA the picked base on human German (Kade train docs). Do not submit to Kaggle until this ranking is stable.
