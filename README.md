This project aims at providing a novel LSTM-based approach for profiling users and monitoring progress in traditional online learning platforms.

==Backend setup==
 ``venv/Scripts/activate``
``pip install -r requirements.txt`'
`uvicorn src.api.main:app --host localhost --port 8000`
Live server for dashboard/index.html

==Test Runs==
``pytest tests/test_api.py -v``
Currently only runs 4 test suites