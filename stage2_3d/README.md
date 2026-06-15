# Stage 2 - 3D view (planned)

Third-person 3D world. Physics becomes 6-DOF (x,y,z + pitch/yaw/roll); the RL
observation/action spaces grow. Tabular Q does not survive this jump -- stage 2 is built
on a deep agent (PPO/SAC).

**This is a fidelity port, not the next capability step.** It happens only *after* the
multi-agent team fight (evader / interceptor / counter-interceptor) is working in the fast
2D engine. We port the matured multi-agent game up to 3D, not the single-agent
interceptor -- multi-agent RL is developed and debugged in 2D first, where iteration is
cheap. Reuses core/'s env interface, per-role reward design, and training scaffolding.

Likely engine: PyBullet (Python-native, headless-friendly, fast for RL). See
`stage3_fpv/README.md` for the camera/segmentation evaluation that also informs stage 2's
engine choice.