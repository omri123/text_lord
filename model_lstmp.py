import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LSTM_LORD(nn.Module):

    def __init__(self, dim, layers, vocab_size, number_of_samples, noise_std):
        super(LSTM_LORD, self).__init__()
        self.dim = dim

        self.word_embeddings = nn.Embedding(vocab_size, dim)

        self.stars_embedding = nn.ModuleList([nn.Embedding(2, dim) for i in range(2 * layers)]) # 2 - one for c, one for h
        self.sample_embedding = nn.ModuleList([nn.Embedding(number_of_samples, dim) for i in range(2 * layers)])

        # the LSTM itself
        self.lstm = nn.LSTM(dim, dim, num_layers=layers, batch_first=True)

        # The linear layer that maps from hidden state space to word space
        self.fc = nn.Linear(dim, vocab_size)
        self.noise = Normal(loc=0.0, scale=noise_std)

    def forward(self, sentences, state):
        # sentences are shifted

        w_embeds = self.word_embeddings(sentences)

        assert state
        #             state = self.create_initial_hiddens(stars, sample_ids)

        #         embeds = w_embeds + s_embeds + id_embeds


        lstm_out, lstm_state = self.lstm(w_embeds, state)
        logits = self.fc(lstm_out)
        # probabilities = F.log_softmax(logits, dim=2)
        return logits, lstm_state

    def create_initial_hiddens(self, stars, sample_ids):
        s_embeds = [elayer(stars).unsqueeze_(0) for elayer in self.stars_embedding]

        id_embeds = [elayer(sample_ids).unsqueeze_(0) for elayer in self.sample_embedding]
        if self.training:
            id_embeds = [embed + self.noise.sample(sample_shape=embed.size()).to(embed.device) for embed in id_embeds]

        joint = [s_embed + id_embed for s_embed, id_embed in zip(s_embeds, id_embeds)]

        h = torch.cat(joint[0 : len(joint) // 2], 0)
        c = torch.cat(joint[len(joint) // 2 : ], 0)
        state = (h, c)
        return state

