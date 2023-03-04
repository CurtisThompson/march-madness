from xgboost import XGBClassifier
from xgboost import plot_tree, plot_importance
import matplotlib.pyplot as plt


def load_models(name='default_model'):
    """Loads separate mens and womens classifier models."""
    men_model = XGBClassifier()
    men_model.load_model(f'./data/models/{name}_men.mdl')
    women_model = XGBClassifier()
    women_model.load_model(f'./data/models/{name}_women.mdl')
    return men_model, women_model


if __name__ == "__main__":
    mmodel, wmodel = load_models()
    plot_tree(mmodel)
    plt.show()