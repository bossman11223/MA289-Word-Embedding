# AI Acknowledgement

**Project: MA289 Neural Network Word Embedding Competition**

Claude Code. Assistance given to the author, AI.
I used Claude Code in three sessions on this project, on 14NOV2026, 20NOV2026, and 21NOV2026.
My main prompts were "whats wrong with this code why am i gettign an error," "can i run this notebook," "do that for me," "commit this to github," and "make visualizations for how loss changes with different parameters"; my other prompts asked what a DAAW citation is and asked the AI to commit this acknowledgement.
In response, the AI ran my test suite and every training configuration without reproducing an error, and it pointed out two errors my project would hit: the window-3 configuration needed a pairs file that `build_pairs.py` had not created yet, and the exploration notebook crashed when no run folder existed.
It then installed `ipykernel` and `nbconvert` into my virtual environment, trained my baseline configuration, ran my exploration notebook on that run, summarized the loss, nearest-neighbor, and cosine results, explained why training is done in the scripts rather than the notebook, and committed and pushed the project to a private GitHub repository.
For the loss visualizations, it wrote two new scripts, `sweep.py` (which changes one setting at a time among embedding size, learning rate, batch size, hidden layer size, and weight decay) and `plot_sweep.py` (which plots loss against each setting); the sweep stopped with an error partway through its 26 runs, so no plots were produced, and neither script is committed.
I used the AI's output to get my existing pipeline running, to produce the baseline submission, and to put the code under version control; it did not change my model design, data handling, cosine-to-score mappings, or choice of experiments.
I uploaded the baseline submission to Kaggle and checked its score myself.
The skip-gram model, the training and submission scripts, the configurations, the tests, and the exploration notebook are my own work, written before these sessions, and the AI helped me debug, run, and save that work without creating it.
The complete conversation, including every prompt I gave and every response the AI returned, is recorded in the accompanying transcript ([Transcript.md](Transcript.md), in this folder).
Anthropic, (https://claude.ai). West Point, NY, 21NOV2026.
