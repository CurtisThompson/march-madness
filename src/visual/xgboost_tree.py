from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt


def load_models(name='default_model'):
    """
    Load separate mens and womens classifier models.
    
    Args:
        name: Name of model to load. String.
    
    Returns:
        Two classifier models, one for mens data and one for womens data.
    """
    men_model = XGBClassifier()
    men_model.load_model(f'./data/models/{name}_men.mdl')
    women_model = XGBClassifier()
    women_model.load_model(f'./data/models/{name}_women.mdl')
    return men_model, women_model


if __name__ == "__main__":
    mmodel, wmodel = load_models()
    plot_tree(mmodel)
    plt.show()