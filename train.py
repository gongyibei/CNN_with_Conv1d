import model
from util import *
from torch import nn,optim

# setting
epochs = 5
vote_rule = 'maj'  # Available rule: 'maj', 'sum'

# initial net
net = CNN1D()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# initial datr
dataset = read_dataset()

# train
for epoch in range(epochs):
    for i, (audio, label) in enumerate():
        clips = split_audio(audio)

        # caculate pred
        preds = []
        for clip in clips:
            preds.append(net(clip))

        if vote_rule == 'maj':
            pred = maj_vote(preds)
        elif vote_rule == 'sum':
            pred = sum_vote(preds)
        else:
            pred = maj_vote(preds)

        # update para
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
