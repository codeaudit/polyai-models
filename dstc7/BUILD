python_library(
    name = "test_reader",
    srcs = ["test_reader.py"],
    deps = [
        "//third_party/python:tensorflow",
        "//utils:gfile_utils",
    ],
)

python_binary(
    name = "evaluate_encoder",
    main = "evaluate_encoder.py",
    deps = [
        ":test_reader",
        "//models:encoder_model_client",
        "//plog",
        "//third_party/python:numpy",
    ],
)

python_test(
    name = "test_reader_test",
    srcs = ["test_reader_test.py"],
    data = [
        "testdata/test_examples.json",
        "testdata/test_labels.tsv",
    ],
    deps = [":test_reader"],
)
