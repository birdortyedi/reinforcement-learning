from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from parameters import *
from Environment import PongEnvironment
import os


class Agent:
    def __init__(self, load_model=False, testing=False):
        self.critic = self.build_critic()
        if CONTINUOUS is False:
            self.actor = self.build_actor()
        else:
            self.actor = self.build_actor_continuous()

        self.env = PongEnvironment()

        self.episode = 0
        self.testing = testing
        if not self.testing:
            self.NUM_EPISODE = EPISODES
        else:
            self.NUM_EPISODE = 100
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.gradient_steps = 0
        self.action_noise = NOISE
        self.load_model = load_model

        if self.load_model:
            self.actor.load_weights("./weights/actor_weights.h5")
            self.critic.load_weights("./weights/critic_weights.h5")

    def build_actor(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[self.proximal_policy_optimization_loss(advantage=advantage,
                                                                   old_prediction=old_prediction)])
        model.summary()

        return model

    def build_actor_continuous(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, name='output', activation='tanh')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[self.proximal_policy_optimization_loss_continuous(advantage=advantage,
                                                                              old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        if self.episode % VALIDATION_EACH == 0:
            self.val = True
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_action_continuous(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        else:
            action = action_matrix = p[0]
        return action, action_matrix, p

    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        """
        Sometimes this rollout exceeds buffer size and thats normal. For example,
        buffer size is 250 but we don't observe any done's until 250. This rollout
        continues until we see a done(either a goal reach or time exceed done)
        This can be altered by counting a variable and checking that variable with
        buffer size.
        """
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        done = False
        untransformed_reward = []
        while len(batch[0]) < BUFFER_SIZE:
            if CONTINUOUS is False:
                action, action_matrix, predicted_action = self.get_action()
            else:
                action, action_matrix, predicted_action = self.get_action_continuous()
            observation, reward, done = self.env.step(action)
            untransformed_reward.append(reward)
            if self.gradient_steps % RENDER_EACH == 0:
                self.env.render()
            self.reward.append(reward)
            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), \
            np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward, untransformed_reward

    def run(self):
        # Note that in PPO, episodes are not counted, instead, we do a rollout of K steps and learn from that
        while self.episode < self.NUM_EPISODE:
            """
            In the original code, these arrays are clipped to BUFFER_SIZE number of elements
            but I found out that this way it performs better so I updated this -Emir
            """
            obs, action, pred, reward, untransformed_reward = self.get_batch()
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            if not self.testing:
                # advantage = (advantage - advantage.mean()) / advantage.std()
                actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE,
                                            shuffle=True, epochs=EPOCHS, verbose=False)
                critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True,
                                              epochs=EPOCHS, verbose=False)
                print("Gradient Update:", self.gradient_steps, " Reward: ", sum(untransformed_reward))
                self.gradient_steps += 1

        if not self.testing:
            self.save_weights("./weights")

    def save_weights(self, fpath):
        self.actor.save_weights(filepath=os.path.join(fpath, "actor_weights.h5"))
        self.critic.save_weights(filepath=os.path.join(fpath, "critic_weights.h5"))

    @staticmethod
    def proximal_policy_optimization_loss(advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING,
                                                           max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
                        prob * K.log(prob + 1e-10)))

        return loss

    @staticmethod
    def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
        def loss(y_true, y_pred):
            var = K.square(NOISE)
            pi = 3.1415926
            denom = K.sqrt(2 * pi * var)
            prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

            prob = prob_num / denom
            old_prob = old_prob_num / denom
            r = prob / (old_prob + 1e-10)

            return -K.mean(K.minimum(r * advantage,
                                     K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))

        return loss
