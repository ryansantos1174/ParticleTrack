import tensorflow as tf
import plotting_functions as plot
import os
import matplotlib.pyplot as plt


def model_trainer(model_values, model_names, x_data, y_data, x_test, y_test,
                  max_values, save=False, save_dir='.',
                  optimizer=tf.keras.optimizers.Adam(), batch_size=32,
                  epochs=100, bad_points=False, dimensions=3):
    if type(model_values) != "list":
        model_name = model_names
        
        #Creates directory to save results to 
        if save:
            os.mkdir(os.path.join(save_dir, model_name))
        save_directory = os.path.join(save_dir, model_name)
        print(save_directory)

        #Compiles and sets up model
        model = model_values
        model.compile(optimizer=optimizer, loss='MSE')
        history = model.fit(x=x_data, y=y_data, batch_size=batch_size,
                            epochs=epochs, validation_split=0.2)
        loss_value = history.history['loss']
        valid_loss = history.history['val_loss']
        print("Evaluate on test data")
        results = model.evaluate(x_test, y_test, batch_size=128)
        print("test loss:", results)

        if save:
            with open(os.path.join(save_directory, 'Evaluation.txt'), "a") as txt:
                txt.write(f'test loss: {results}')

        #Plot Loss Graph
        plt.title(f'{model_name}: Plot Loss')
        plt.plot(range(epochs), loss_value)
        plt.plot(range(epochs), valid_loss)
        if save:
            plot_loss_savefile = os.path.join(save_directory, 'Plot_Loss.png')
            plt.savefig(plot_loss_savefile)
        plt.show()
        
        #Plots tracks and residuals as defined in functions.py
        plot.Plot_All_Results(model, x_test, y_test, max_values, SAVE=save, dimension=dimensions,
                              save_dir=save_directory)

    else:
        for index, model in enumerate(model_values):
            if save:
                os.mkdir(os.path.join(save_dir, model_names[index]))
            save_directory = os.path.join(save_dir, model_names[index])
            model = model_values[index]
            model.compile(optimizer=optimizer, loss='MSE')
            history = model.fit(x=x_data, y=y_data, batch_size=batch_size,
                                epochs=epochs, validation_split=0.2)
            loss_value = history.history['loss']
            valid_loss = history.history['val_loss']

            print("Evaluate on test data")
            results = model.evaluate(x_test, y_test, batch_size=128)
            print("test loss", results)
            if save:
                with open(os.path.join(save_directory, 'Evaluation.txt')) as txt:
                    txt.write(f'test loss: {results}')

            plt.title(f'{model}: Plot Loss')
            plt.plot(range(epochs), loss_value)
            plt.plot(range(epochs), valid_loss)
            if save:
                plt.savefig(os.path.join(save_directory, 'Plot_Loss.png'))
            plt.show()

            plot.Plot_All_Results(model, x_test, y_test, max_values, SAVE=save,
                                  save_dir=save_directory, dimension=dimensions, bad_points=bad_points)
