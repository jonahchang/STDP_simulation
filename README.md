
# STDP Simulation

## Overview
This project is a simulation of Spike-Timing-Dependent Plasticity (STDP) in neural networks. The simulation models the behavior of neurons and synapses, illustrating how synaptic strengths are adjusted based on the timing of spikes between neurons. This project is particularly useful for understanding the mechanisms behind learning and memory in the brain.

## Features
- Simulation of leaky integrate-and-fire neurons
- Implementation of STDP rules for synaptic plasticity
- Visualization of neuronal spikes and synaptic changes
- Customizable parameters for neuron and synapse properties
- Comprehensive documentation and example configurations

## Installation
To get started with the STDP Simulation project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/STDP_simulation.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd STDP_simulation
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the simulation, use the following command:
```bash
python main.py
```
You can customize the simulation parameters by editing the `config.json` file.

## Project Structure
- `main.py`: The main script to run the simulation.
- `neuron.py`: Contains the Neuron class, which models the behavior of a single neuron.
- `synapse.py`: Contains the Synapse class, which models the connections between neurons.
- `config.json`: Configuration file for setting simulation parameters.
- `README.md`: Project documentation.

## Contributions
Contributions to the project are welcome. Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For any questions or inquiries, please contact Jonah Chang at jonahpchang@gmail.com.
