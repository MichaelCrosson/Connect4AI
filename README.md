# Connect 4 AI Bot Project

## Overview
This project combines neural networks, Monte Carlo Tree Search (MCTS), cloud hosting, and interactive web development to build an AI bot that plays Connect 4. Users can interact with the bot through a web-based interface, play games, and analyze the bot's decision-making process. Project Link: [https://lazy-similar-sparrow.anvil.app](#)

## Key Features
- **Data Generation:** Used MCTS to generate a large, diverse dataset of Connect 4 board states and corresponding optimal moves.
- **Model Training:** Developed two machine learning models for decision-making:
  - **Convolutional Neural Network (CNN)** for image-based classification of moves.
  - **Transformer model** for sequence-based classification of moves.
- **Interactive Web Application:** Built a user-friendly web application using Anvil, allowing users to:
  - Play Connect 4 against the trained CNN or Transformer bot.
  - Analyze model performance on different board states.
  - View details about the training process and model architecture.
- **Cloud Hosting:** Deployed the backend using AWS Lightsail and Docker, ensuring high availability and compatibility across platforms.

## Implementation Details

### 1. Data Generation
- **Monte Carlo Tree Search (MCTS):**
  - Played self-games to simulate optimal moves.
  - Incorporated randomness for diverse datasets.
  - Dataset structure:
    - **Features (X):** Board states.
    - **Labels (Y):** Recommended moves.

### 2. Neural Networks
#### CNN Architecture:
- Multiple convolutional layers with varying filter sizes.
- Max-pooling layers for dimensionality reduction.
- Dense layers with dropout for regularization.
- Trained using GPUs on Google Colab.

#### Transformer Architecture:
- Multi-Head Self-Attention layers.
- Positional encoding for board state representation.
- Dense layers for classification.
- Tuned hyperparameters, including head count, hidden dimensions, and layer depth.

### 3. Interactive Webpage
#### Features:
- **Login System:** Secure access with pre-created admin account.
- **Training Analysis:** Visualizations of board states, model predictions, and performance.
- **Game Interface:** Playable Connect 4 game with bot opponents.
  - Option to select CNN or Transformer bot.
  - New game functionality.

#### Deployment:
- Hosted front-end on Anvil.
- Backend API powered by TensorFlow and AWS.

### 4. Cloud Hosting
- **AWS Lightsail:** Deployed backend using Docker for compatibility.
- Used Anvil uplink for communication between the web app and backend models.
- Maintained uptime for grading period.

## Installation and Usage

### Prerequisites
- Python 3.9+
- Docker
- AWS Lightsail account
- Anvil account

### Clone the repository:
   ```bash
   git clone https://github.com/MichaelCrosson/Connect4AI.git
   cd Connect4AI
   ```

## Usage
- Visit the hosted web app: [Connect4AI.anvil.app](#)
- Login using:
  - **Email:** dan
  - **Password:** Optimization1234
- Play against the bot and view detailed training analysis.

## References
- [Connect 4 Rules](https://en.wikipedia.org/wiki/Connect_Four)
- [Monte Carlo Tree Search Resources](https://mcts.netlify.app/)
- [Anvil Documentation](https://anvil.works/docs/)
- [AWS Lightsail](https://lightsail.aws.amazon.com/)

