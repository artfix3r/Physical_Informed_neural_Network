### Neural ODEs Background

The concept originates from Lagaris et al (1998), proposing a method to solve differential equations using neural networks. The neural network represents the solution and is trained to satisfy the differential equation's conditions.

#### Problem Setup

Consider a system of ordinary differential equations:

$u' = f(u, t)$

with $t \in [0, 1]$ and initial condition $u(0) = u_0 $

#### Neural Network Approximation

We approximate the solution using a neural network:

$\text{NN}(t) \approx u(t)$ 

If $\text{NN}(t)$  were the true solution, it would satisfy:

$\text{NN}'(t) = f(\text{NN}(t), t)$ 

#### Loss Function

This leads to our initial loss function:

$L(\omega) = \sum_i \left(\frac{d\text{NN}(t_i)}{dt} - f(\text{NN}(t_i), t_i)\right)^2$ 

where:
- \(t_i\) can be chosen randomly or as a grid
- When minimized using gradient descent with automatic differentiation, we get $$\frac{d\text{NN}(t)}{dt} \approx f(\text{NN}(t_i), t_i)$$


#### Initial Condition Handling

To incorporate the initial condition, we add a term to the loss function:

$L(\omega) = \sum_i \left(\frac{d\text{NN}(t_i)}{dt} - f(\text{NN}(t_i), t_i)\right)^2 + (\text{NN}(0) - u_0)^2$ 

The parameters $\omega$ define the neural network NN that approximates $ u $. The problem reduces to finding weights that minimize this loss function.


### ODE Example

#### Problem Statement
Let's look at the ODE:

$$\frac{du}{dt} = \cos 2\pi t$$

Initial condition:

$$u(0) = 1$$

The exact solution:

$$u(t) = \frac{1}{2\pi}\sin 2\pi t + 1$$

#### Neural Network Implementation

The neural network should approximate this solution. Following Lagaris et al (1998), we define:

$$NN(t) \approx u(t)$$

#### Loss Function
The loss function combines two terms:

1. The differential equation constraint:
$$L_{DE}(\omega) = \sum_i \left(\frac{dNN(t_i)}{dt} - \cos(2\pi t_i)\right)^2$$

2. The initial condition constraint:
$$L_{IC}(\omega) = (NN(0) - 1)^2$$

The total loss function is:
$$L(\omega) = L_{DE}(\omega) + L_{IC}(\omega)$$

where $\omega$ represents the neural network parameters that need to be optimized to minimize this loss function.




### Loss Functions

#### ODE Loss
The loss for the differential equation:

$$L_{ODE} = \frac{1}{n}\sum_{i=1}^n \left(\frac{dNN(t_i)}{dt} - \cos 2\pi t_i\right)^2$$

#### Initial Condition Loss
The loss for the initial condition:

$$L_{IC} = \frac{1}{n}\sum_{i=1}^n (NN(0) - 1)^2$$

#### Total Loss
The combined loss function:

$$L_{Total} = L_{ODE} + L_{IC}$$

## Result 
![image](https://github.com/user-attachments/assets/a027d5d5-d6b8-472f-8073-d612c366c7f6)


