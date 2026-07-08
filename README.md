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

## Web App
Trained models are connected to a web app that allows users to play against the trained model. Currently only the tic-tac-toe model is supported.  
To run the web app locally, navigate to the root directory and run `uvicorn webapp.api:app --reload`.
The web app is also hosted on Render, at https://erics-ai-portfolio.onrender.com/, where trained models are available for play and deployed through Google Cloud Storage.

## Future Work
- Connect 4
- 5-In-A-Row
- Stochastic game settings