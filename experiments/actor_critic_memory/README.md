Actor Critic Memory

Idea comes BIML suppose actor takes in state and memory m, where m is the episode starting from the closest state to the current one

3 Experiments to do : 

1. Get KNN states, input the states rewards and actions selected up to 50 next states if less in episode then repeat the terminal state Inlcude similarity score
2. Get KNN states, like in VISR input cumulative discounted future states µ and rewards
3. Something involving LSTMs

Thoughts behind Idea : 
Firstly, reduce sample inneficiency by letting the model almost have a lookup table that is bootstrapped through the KNN. When seeing a similar state it can see what it did later
and eventually see a way out sort of like Alphazero in a sense but it won't be a tree search just a memory
Secondly, for actor critic specifically the model can see Values predicted of states, exploration taken and how it led to higher rewards.  
