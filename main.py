if __name__ == "__main__" :
    from model import *
    quit = False
    iterations = 0
    
    nn = init_model()

    while not quit:
        print("1. Train model")
        print("2. Test model")
        print("3. Quit and save")
        print("Enter your choice: ")
        choice = int(input())
        if choice == 1:
            print("Enter number of iterations: ")
            iterations = int(input())
            print("Training model...")
            gradient_descent(X_train, Y_train, 0.10, iterations, nn)
            iterations = 0
        elif choice == 2:
            print("how many tests")
            iterations = int(input())
            print("Testing model...")
            for i in range(iterations):
                test_prediction(i, nn)

        elif choice == 3:
            save(nn)
            quit = True
        else:
            print("Invalid choice. Enter 1-3")