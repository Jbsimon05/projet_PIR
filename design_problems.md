# General design problems





## in rrt_star_snamo


- The cost calculation is wrong for S-NAMO and there is no exit without finding the actual goal configuration


- the possible configs have a discrete maximum because of the steer() function


- the hashing is bad (allows for collisions because of the discretization of configs), this is important because of data the agent needs from the search


- using a different map from the rest of the upstream agent, this creates issues with collision detection later





## For a generic implementation


- how to handle collisions generically ? the center of rotation isn't necessarily the robot (think large obstacle/future needs where rotation is done against something)


- self.steer, self.random_pose, and the cost calculation MUST be provided by the upstream implementation


    - self.steer :


        - steering capability (both rotation and movement towards the objective) is dictated by the robot using the algorithm, the algorithm cannot assume how the robot can move


        - steering must be handled by the robot in order to put the right structures in the tree


    - self.random_pose


        - to generate the correct structures to use for the rest of an iteration


        - if everything is a PoseModel, this doesn't have to be generic, but the generic A* implementation is even generic on the start and goal structures, this would be a genericity regression and it would impose types on steer() and the node cost calculation


        - problem : this makes generic informed RRT* impossible with the current code structure


    - node cost calculation :


        - to allow for different kinds of heuristics and path objectives





- what should it return ?


    - the raw exploration tree


## What's left
- make a visualization in rviz2 for the RRT* path if found and the tree
- removing/replacing find_best_... 
    - look at is_there_opening 
    - get the tree up to ~500 (or N) nodes, and do that check on those nodes
    - search for the first possible solution or cap it at ~2000 after a time
- making the goal_tolerance small
- making collision detection better
- making sure at the end of the path that the end node leaves enough space to go back
- paramétrer le type de robot : "DiffDrive" ou "Holonomic"