ai-project/
│
├── data/                  # dữ liệu
│   ├── raw/              # video gốc
│   ├── processed/        # đã extract feature
│   ├── annotations/      # labels.csv
│
├── src/                  # source code chính
│   ├── data/             # xử lý data
│   ├── features/         # extract feature (MediaPipe)
│   ├── models/           # model (LSTM, Transformer)
│   ├── training/         # train loop
│   ├── evaluation/       # test / metrics
│   ├── utils/            # helper functions
│
├── notebooks/            # Jupyter test nhanh
│
├── configs/              # config training
│
├── outputs/              # kết quả
│   ├── models/           # saved model (.pt)
│   ├── logs/             # training logs
│   ├── predictions/      # output
│
├── scripts/              # chạy nhanh (CLI)
│
├── requirements.txt
├── README.md