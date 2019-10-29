[![PolyAI](polyai-logo.png)](https://poly-ai.com/)

# polyai-models

Neural Models for Conversational AI.



## Development

Setting up an environment for development:

* Create a python 3 virtual environment

```bash
python3 -m venv ./venv
```

* Install the requirements

```bash
. venv/bin/activate
pip install -r requirements.txt
```

* Run the unit tests

```bash
python -m unittest discover -p '*_test.py' .
```

Pull requests will trigger a CircleCI build that:

* runs flake8 and isort
* runs the unit tests
