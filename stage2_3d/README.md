# Stage 2 - 3D view (planned)
Third-person 3D world (three.js). Physics becomes 6-DOF (x,y,z + pitch/yaw/roll);
the RL observation/action spaces grow. Tabular Q does not survive this jump --
build stage 2 on a deep agent (DQN/PPO). Reuses core/'s env interface, reward
philosophy, and training scaffolding.
