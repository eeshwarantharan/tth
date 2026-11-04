# Why two Coder 1.5B files with the same label score differently

100 held-out shell tasks, same settings (context 2048, ubatch 128, no checkpoints), one seed, greedy decoding.

| Build | Template in the prompt | Correct |
|---|---|---|
| Official Qwen Q4_K_M | single brace | 17 |
| Official Qwen Q2_K | doubled brace (its own) | 19 |
| Official Qwen Q2_K | single brace | 20 |
| bartowski Q4_K_M | single brace (its own) | 33 |
| bartowski IQ3_XS | doubled brace | 40 |
| bartowski IQ3_XS | single brace (its own) | 40 |

- The template isn't the cause: swapping it changed each build by at most one task.
- The file is: bartowski Q4_K_M beats official Q4_K_M by 16 (95% paired interval +5 to +26).
- Lower precision didn't visibly hurt: bartowski IQ3_XS minus Q4_K_M is +7 (-3 to +17), so don't read this as IQ3_XS being better.
- The official file has 339 tensors and 1.78B parameters. bartowski's has 338 and 1.54B, and was quantized with an importance matrix. These two differences aren't separated, so **the cause within the file is unknown**.

This suite also picked the winner, so 40 is somewhat optimistic. Later claims need tasks that played no part in the choice.
