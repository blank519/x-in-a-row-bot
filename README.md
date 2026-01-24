# x-in-a-row-bot
Using Reinforcement Learning to train a model to play games like Tic-Tac-Toe (3-in-a-row), Connect 4 (4-in-a-row), and 5-In-A-Row.  
Uses PettingZoo and Stable Baselines3 for training.  

## Tic-Tac-Toe Demo  
An example of a trained model playing Tic-Tac-Toe against itself.  

<img src="docs/tictactoe-demo.gif" width="640" />  

Training methodology:  
1. **Warmup**: Trains against an opponent policy that selects random moves.
2. **Self-Play With Opponent Pool**: Trains against a pool of opponents that includes the random move policy, a heuristic policy, and snapshots of previous trained models.
3. **(Optional) Fine-tuning**: Continued training similar to step 2, but with modified hyperparameters and a lower likelihood of training against the random move policy. This stage is unnecessary for a game as simple as Tic-Tac-Toe.

## Future Work
- Connect 4
- 5-In-A-Row
- Fire Emblem-style turn-based strategy (separate project)