from model.discriminative import OneVsOne, OneVsTheRest, MultipleClass, Fisher
from model.generative import Bayesian

def main():
    model = Fisher("data/iris.csv")
    model.train()
    model.print_evaluation()
    
if __name__ == "__main__":
    main()