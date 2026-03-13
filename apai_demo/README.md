## APAI Proof of Concept

save the responses to apai_responses.txt, and generate capability trajectory
plots in apai_simulation.pdf and apai_simulation.png.
Install dependencies:
    pip install google-generativeai matplotlib scipy numpy

Set your Gemini API key:
    export GEMINI_API_KEY=your_key_here

Run:
    python main.py

This will query the Google Gemini LLM (gemini-1.5-flash) with the same Markov chain
problem under three assistance levels (full assistance, partial, full deferral),
save the responses to apai_responses.txt, and generate capability trajectory
plots in apai_simulation.pdf and apai_simulation.png.
