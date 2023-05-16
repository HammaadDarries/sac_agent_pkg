import torch

class SACAgent:
    def __init__(self, filepath):
        self.actor = torch.load(filepath)

    def act(self, state):
        state = torch.FloatTensor(state)
        action, log_prob = self.actor(state)
        
        return action.detach().numpy()