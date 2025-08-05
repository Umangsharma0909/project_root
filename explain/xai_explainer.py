import shap

class XAIExplainer:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.explainer = shap.KernelExplainer(self.model.predict, self.sample_background())

    def sample_background(self, n=100):
        samples = []
        for _ in range(n):
            obs, _ = self.env.reset()
            samples.append(obs)
        return np.array(samples)

    def explain(self, observation):
        shap_values = self.explainer.shap_values(observation)
        shap.summary_plot(shap_values, observation)
