"""Core abstractions for musicbench.

The pipeline is:

    Dataset -> [Sample]  -> ModelAdapter -> [Prediction]
                              |
                              v
                          Metric(s) -> metrics dict

    Runner orchestrates: load config -> load dataset -> load adapter ->
    predict -> serialize predictions -> compute metrics -> write artifacts.
"""
