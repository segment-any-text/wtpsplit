import torch

torch.save(
    {
        "language_code": {
            "sentence": {
                "dummy-dataset": {
                    "meta": {
                        "train_data": ["train sentence 1", "train sentence 2"],
                    },
                    "data": [
                        "train sentence 1", "train sentence 2"
                    ]
                }
            }
        }
    },
    "dummy-dataset.pth"
)