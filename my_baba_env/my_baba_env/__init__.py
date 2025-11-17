from gymnasium.envs.registration import register

register(
    id="BabaIsYou-v1",
    entry_point="my_baba_env.envs:BabaIsYouEnv",
)
