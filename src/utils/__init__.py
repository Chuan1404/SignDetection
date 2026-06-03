from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    features = []
    texts = []

    for feature, text in batch:
        features.append(feature)
        texts.append(text)

    features = pad_sequence(
        features,
        batch_first=True
    )

    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    labels = texts.clone()

    labels[
        labels == tokenizer.pad_token_id
    ] = -100

    return features, labels

def compute_loss(
    hand_features,
    labels
):

    hand_features = hand_features.to(
        DEVICE,
        non_blocking=True
    )

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    with torch.cuda.amp.autocast():

        outputs = model(
            hand_features,
            labels=labels
        )

        loss = outputs.loss

    return loss