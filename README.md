# Pong-Game-DQN
Analysis of Deep Q-Network algorithm on Simple Pong Environment

## State Representation
I defined agent’s states like the following:
- Position of the left paddle on the y-axis
- Position of the right paddle on the y-axis
- Position of the ball on the y-axis
- Position of the ball on the x-axis
- Velocity of the ball on the x-direction
- Velocity of the ball on the y-direction

## Reward Function Definition
A set of predefined constants delineate the rewards and penalties for different in-game events as following:
1. Game End Condition: If the game concludes , the function checks the outcome: - If the agent scores, the +10 reward is granted.
- If the opponent scores, the -10 penalty is deducted.
2. In-Game Rewards/Penalties: For ongoing games:
- The function first checks for the ball's collision with the agent's paddle. If the ball hits the center of the paddle, a reward of +0.1 is given; otherwise, a penalty of -0.1 is applied.
- If there's no collision, the function evaluates the agent's movement towards or away from the ball, using the difference in vertical distance between the ball and the paddle. Depending on the movement direction, a corresponding reward of +0.5 or penalty of - 0.5 is assigned.

## Results

<img width="563" alt="Screenshot 2023-08-31 at 17 33 19" src="https://github.com/sinanutkuulu/Pong-Game-DQN-PPO/assets/92628109/bf6f6745-4b29-49f0-8876-51d227cd42e4">

Test of agent (DQN) against nominal player

<img width="836" alt="Screenshot 2023-08-31 at 17 30 04" src="https://github.com/sinanutkuulu/Pong-Game-DQN-PPO/assets/92628109/77d52ef6-c84f-4516-953c-216d4dc12268">
