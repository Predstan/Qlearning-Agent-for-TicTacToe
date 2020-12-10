



from collections import defaultdict
import random
import math
import numpy as np


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly. 
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.actionable = legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
    def get_table(self):
      return self._qvalues
      
    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.actionable.get_legal_actions(state=state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        max_value = float("-inf")
        for action in possible_actions:
          action_value = self.get_qvalue(state, action)
          if action_value > max_value:
            max_value = action_value

        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha
        Q_value = self.get_qvalue(state, action)
        value = self.get_value(next_state)
        Q_value = ((1 - learning_rate) * Q_value) + (learning_rate * (reward + (gamma * value)))

        self.set_qvalue(state, action, Q_value )

    def get_best_action(self, state):
        import operator
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = self.actionable.get_legal_actions(state= state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None
        
        all_actions = {action :self.get_qvalue(state, action) for action in possible_actions}


        return max(all_actions.items(), key=operator.itemgetter(1))[0]

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list). 
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.actionable.get_legal_actions(state = state)
        action = None

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None
  
        # agent parameters:
        epsilon = self.epsilon
        best_action = self.get_best_action(state)
        random_choice = np.random.choice(possible_actions)
        chosen_action = np.random.choice([best_action, random_choice], p = [1-self.epsilon, self.epsilon]   )
        return best_action