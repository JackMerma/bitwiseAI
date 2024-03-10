# Bitwise AI

This project uses the PyTorch library to train basic bitwise operations (**XOR, AND, OR, and NOT**). The process begins by using the `training_*` scripts for training the four models and creating image outputs, which will be explained later. Then, the validation script is used, and finally, you can employ it in the main function for processing real information with the defined functions in `model_source`. Now, I'm going to explain:

## Training Phase

### training_source

In the **training_source.py** file, there is a simple classifier class that uses a PyTorch module (`nn.Module`) to create a neural network module that employs **one layer** to process the input. You can also define the number of input variables, hidden neurons, and the output number. Additionally, this function defines a forward function to specify the processing order. It is important to note that this module uses the **hyperbolic tangent** (Tanh) activation function.

There are also four classes that use a defined `Dataset` function, utilizing the `data.Dataset` function to generate random data and their results (labels). The only difference between these classes is the processing of the label and the number of inputs (for example, the `NOTDatabase` only uses one input). It is also important to highlight the **Gaussian noise** used by all these classes on the binary inputs. Finally, there are some functions that will process different phases of this project:

+ `train_model`: Trains the model using an independent model, an optimizer, a data loader (training inputs), a loss module to process the loss, an epoch number, and a model name. In each epoch, the number of data will be processed with a batch number defined in the data loader object. Finally, in each iteration, the data is trained with five basic steps:
  1. Moving input data to the device (only if you use GPU).
  2. Running the model on the input data.
  3. Calculating the loss.
  4. Performing backpropagation.
  5. Updating parameters.

   ```python
    def train_model(model, optimizer, data_loader, loss_module, num_epochs=100, name_model="Operation"):
        model.train()

        for epoch in tqdm(range(num_epochs), desc=f"{name_model} model"):
            for data_inputs, data_labels in data_loader:

                # step 1: move input data to device (just if we use GPU)
                # data_inputs = data_inputs.to(device)
                # data_labels = data_labels.to(device)

                # step 2: run the model on the input data
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1)

                # step 3: calculate the loss
                loss = loss_module(preds, data_labels.float())

                # step 4: perform backpropagation
                optimizer.zero_grad()
                loss.backward()

                # step 5: update parameters
                optimizer.step()
    ```

+ `load_models`: Loads previously saved models in the **models** folder.
+ `eval_model`: Implemented function to evaluate the model using a testing dataset. For this, the function only calculates the **accuracy** value. It is important to indicate that the model output uses the **sigmoid** function to obtain a value between 0 and 1 to normalize the result (a float value). Finally, it counts the TP, TN, FP, and FN to calculate the accuracy.

### The plots/ folder

In this folder, we save the created datasets shown as images using the `matplotlib.pyplot` library and also display the result of the evaluation phase, showing the area of the binary classes (0 and 1 outputs). I recommend reviewing these images.

### The models/ folder

This folder saves the trained models with the `.tar` extension.

### training_model

The **training_model.py** file loads the datasets for each operator (could be binary or unary), the loss module, and the optimizer. Finally, it trains the model and saves them in the **model/** folder. Also, in this process, the script creates and saves the dataset images and the testing dataset classification in the **plots/** folder.

## Evaluation phase

### evaluating_model

This file creates a testing dataset using the classes already seen and also uses the `eval_model` function to evaluate the trained model.

## Use of the trained and validated models

The **model_source** file loads the trained models and implements six functions where the first four use the models directly to apply the **XOR, AND, OR, and NOT** operators. The last two use these functions for the **Implication** function (IMP) and the **Biconditional** function (BIC). All of these functions use an `evaluate_operation` function to load the model and get an output using the **sigmoid** function during the evaluation phase/function.

```python
def XOR(input1, input2):
    data = torch.tensor([input1, input2], dtype=torch.float32)
    return evaluate_operation(xor_model, data)

def AND(input1, input2):
    data = torch.tensor([input1, input2], dtype=torch.float32)
    return evaluate_operation(and_model, data)

def OR(input1, input2):
    data = torch.tensor([input1, input2], dtype=torch.float32)
    return evaluate_operation(or_model, data)

def NOT(input1):
    data = torch.tensor([input1],

 dtype=torch.float32)
    return evaluate_operation(not_model, data)

def IMP(input1, input2):
    return OR(NOT(input1), input2)

def BIC(input1, input2):
    return XOR(IMP(input1, input2), IMP(input2, input1))
```

Finally, in the **main.py** file, you can experiment using all of these functions, and this file also shows examples. Keep in mind that you **are not** restricted to using just the 0 or the value, but they are decimally close to these; for example, instead of 0, you can use -0.0001, or instead of 1, you can use 0.92.
