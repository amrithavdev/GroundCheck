\# GroundCheck



\*\*An evidence-based verification layer for AI-extracted information.\*\*



GroundCheck checks whether extracted field values are supported by a source message and whether they match the intended meaning of each field.



For example, given:



> Your account was debited ₹1,250. Your available balance is ₹8,400.



GroundCheck should distinguish the transaction amount (`₹1,250`) from the available balance (`₹8,400`) rather than treating the presence of a number as sufficient evidence.



\## Features



\- Field-level verification of extracted claims.

\- Three verdicts: `SUPPORTED`, `UNSUPPORTED`, and `AMBIGUOUS`.

\- Evidence and explanations linked to the original source text.

\- A rule-based verification engine independent of the demo interface.

\- Optional local AI extraction using Ollama and Gemma 3 1B.

\- Synthetic benchmark, model evaluation, and automated tests.

\- Lightweight browser demo using Python and HTML/CSS/JavaScript.



\## How it works



1\. Provide a source message.

2\. Extract structured claims, optionally using a pretrained model.

3\. Check each value against its source evidence and semantic role.

4\. Return a verdict and explanation.

5\. Evaluate behavior using synthetic test cases.



GroundCheck verifies support within the provided text. It does \*\*not\*\* establish whether the real-world event described by that text is true.



\## Requirements



\- Python 3.10 or compatible Python version.

\- Git.

\- For local AI extraction: Ollama and the `gemma3:1b` model.

\- A modern web browser.



The rule-based verification tests can run without downloading or running the AI model.



\## Run locally



From the project root, create and activate a virtual environment:



```powershell

py -3.10 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

python -m pip install pytest

```



Start the demo:



```powershell

py -3.10 -m webui.server

```



Open `http://127.0.0.1:8000` in your browser.



Run tests:



```powershell

py -3.10 -m pytest -v

```



\## Evaluation



The repository includes a synthetic evaluation dataset and a rule-based benchmark. Results are limited to the included examples and do not establish real-world accuracy.



Small sample sizes produce substantial uncertainty. In particular, high observed precision does not guarantee that false acceptances will not occur in other data.



\## Limitations



\- Evaluation data is synthetic and limited in size.

\- Performance may differ on other message styles, languages, and domains.

\- Ambiguous or unsupported information may be withheld instead of corrected.

\- AI extraction can still make mistakes.

\- Evidence from a source does not prove that the source itself is truthful.



\## Open source



GroundCheck is intended as a reusable Python project. See the repository's `LICENSE` file for the project's licence terms.



Contributions, bug reports, and reproducible test cases are welcome.
