class ExperienceReplayBuffer(object):

    # Saves a trajectory to the buffer
    def save_trajectory(self, state, action, reward, next_state, terminal):
        pass
    
    # Returns a batch for training from the buffer
    def get_train_batch(self):
        batch = []
        return batch
