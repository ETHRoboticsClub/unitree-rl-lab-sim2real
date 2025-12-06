import sys
import os

# Add source directory to python path
sys.path.append("/Users/ethanboren/Desktop/Unif/Zurich/Robotics/Unitree/unitree-rl-lab-sim2real/source/unitree_rl_lab")

print("Importing Go2StandEnvCfg...")
try:
    from unitree_rl_lab.tasks.locomotion.robots.go2.stand_env_cfg import RobotEnvCfg as Go2StandEnvCfg
    print("Successfully imported Go2StandEnvCfg.")
except Exception as e:
    print(f"Failed to import Go2StandEnvCfg: {e}")
    sys.exit(1)

print("Importing Go2StandPolicy...")
try:
    from unitree_rl_lab.tasks.locomotion.policies.go2_stand_policy import Go2StandPolicy
    print("Successfully imported Go2StandPolicy.")
except Exception as e:
    print(f"Failed to import Go2StandPolicy: {e}")
    sys.exit(1)

print("Instantiating Go2StandEnvCfg...")
try:
    cfg = Go2StandEnvCfg()
    print("Successfully instantiated Go2StandEnvCfg.")
except Exception as e:
    print(f"Failed to instantiate Go2StandEnvCfg: {e}")
    # Don't exit here, might be due to missing Isaac Lab context but we want to check policy too.

print("Instantiating Go2StandPolicy...")
try:
    # Random dims for test
    policy = Go2StandPolicy(obs_dim=48, action_dim=12)
    print("Successfully instantiated Go2StandPolicy.")
except Exception as e:
    print(f"Failed to instantiate Go2StandPolicy: {e}")
    sys.exit(1)

print("Verification complete.")
